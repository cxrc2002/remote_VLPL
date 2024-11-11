import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  
        with autocast(enabled=True):  
            x = x.to(torch.float32)  
            ret = super().forward(x)  
        return ret.to(orig_type)  
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Prompt_Argument(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.n_ctx = cfg.TRAINER.VLPL.N_CTX
        half_nctx = self.n_ctx
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.num_heads = 8

        self.gal_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.ctx_dim, kernel_size=256//half_nctx, stride=256//half_nctx, bias=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=self.ctx_dim, kernel_size=512//half_nctx, stride=512//half_nctx, bias=False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=self.ctx_dim, kernel_size=1024//half_nctx, stride=1024//half_nctx, bias=False)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=self.ctx_dim, kernel_size=2048//half_nctx, stride=2048//half_nctx, bias=False)

        self.positional_embedding = nn.Parameter(torch.randn(half_nctx, self.ctx_dim) / self.ctx_dim ** 0.5)
        self.k_proj = nn.Linear(self.ctx_dim, self.ctx_dim)
        self.q_proj = nn.Linear(self.ctx_dim, self.ctx_dim)
        self.v_proj = nn.Linear(self.ctx_dim, self.ctx_dim)
        self.c_proj = nn.Linear(self.ctx_dim, self.ctx_dim)

        self.inputln = LayerNorm(self.ctx_dim)
        self.hin_ln = LayerNorm(self.ctx_dim)
        self.hout_ln = LayerNorm(self.ctx_dim)
        self.output_ln = LayerNorm(self.ctx_dim)
        scale = self.ctx_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(self.ctx_dim, self.ctx_dim))


        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.ctx_dim, self.ctx_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.ctx_dim * 4, self.ctx_dim))
        ]))
        
        self.init_params()

    def init_params(self):
        std = self.c_proj.in_features ** -0.5
        proj_std = (2*self.c_proj.in_features) ** -0.5
        nn.init.normal_(self.q_proj.weight, std=std)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)
        nn.init.normal_(self.mlp.c_fc.weight, std=std)
        nn.init.normal_(self.mlp.c_proj.weight, std=proj_std)
       

    def forward(self, x):
        """
            resnet_input:(n, 256, 56, 56) or (n, 512, 28, 28) or (n, 1024, 14, 14) or (n, 2048, 7, 7)
    
    
        """
        x = self.gal_pool(x)
        # (n, channels, 1, 1)  
        x = x.squeeze(-1).permute(0, 2, 1)
        # (n, 1, channels,)
        if  x.size(2) == 256:
            x = self.conv1(x)
        elif x.size(2) == 512:
            x = self.conv2(x)
        elif x.size(2) == 1024:
            x = self.conv3(x)
        elif x.size(2) == 2048:
            x = self.conv4(x)
        else:
            raise Exception("Something went wrong!")
        # x.shape (n, ctx_dim, 8)

        x = x.permute(2, 0, 1) # (8, n, ctx_dim) # NLD -> LND
        x = x + self.positional_embedding[:, None, :].to(x.dtype).to(x.device)
        x = self.inputln(x)

        h, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        h = x + self.hin_ln(h)

        output = h + self.mlp(self.hout_ln(h))
        output = self.output_ln(output)

        output = output.permute(1, 0, 2)     # LND -> NLD  
        
        if self.proj is not None:
            output = output @ self.proj
        #print(x2.shape)   
         
        return output



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VLPL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        print("Initializing a generic context")
        prompt_prefix = " ".join(["X"] * n_ctx)
        ctx_vectors = torch.empty(4, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, seq_len)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = cfg.TRAINER.VLPL.CLASS_TOKEN_POSITION
        
        
        # self.ctx (4, n_ctx=8, ctx_dim)

    def forward(self, out):
        """
        out is (4, n_ctx//2, dim)

        """

        ctx = self.ctx #(4, n_ctx//2, dim)
        # ctx = torch.cat([out, ctx], dim=1)
        ctx = ctx + out
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        # print(ctx.shape)

        prefix = self.token_prefix.unsqueeze(1).expand(-1, ctx.size(1), -1, -1)
        suffix = self.token_suffix.unsqueeze(1).expand(-1, ctx.size(1), -1, -1)
        # print(suffix)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 4, 1, dim)
                    ctx,     # (n_cls, 4, n_ctx, dim)
                    suffix,  # (n_cls, 4, *, dim)
                ],
                dim=2,
            )
        else:
        
            raise ValueError
        # print(prompts.shape)
        # prompts (n_cls, 4, *, dim)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_argument = Prompt_Argument(cfg, clip_model).to(clip_model.dtype)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features, out1, out2, out3, out4 = self.image_encoder(image.type(self.dtype))  #(n, outdim), (n, 256, 56, 56)..
        out1 = self.prompt_argument(out1)  
        out2 = self.prompt_argument(out2)
        out3 = self.prompt_argument(out3)
        out4 = self.prompt_argument(out4) # (n, 8, ctx_dim)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = []
        for i in range(image_features.size(0)):

            out = torch.stack([out1[i,:,:], out2[i,:,:], out3[i,:,:], out4[i,:,:]], dim=0)
            prompts = self.prompt_learner(out) # prompts (n_cls, 4, *, dim)
            prompts = prompts.reshape((prompts.size(0)*prompts.size(1), prompts.size(2), prompts.size(3)))

            tokenized_prompts = self.tokenized_prompts #(n_cls, *, )
            tokenized_prompts = tokenized_prompts.unsqueeze(1).expand(-1, 4, -1)
            tokenized_prompts = tokenized_prompts.reshape((tokenized_prompts.size(0)*tokenized_prompts.size(1),tokenized_prompts.size(2)))  # (n_cls*4, *)
            # print(tokenized_prompts.shape)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.reshape((text_features.size(0)//4, 4, text_features.size(1)))  # (n_cls, 4, dim)
 
            logit = torch.einsum("d,kmd-> km", image_features[i, :], text_features)
            logit = logit*logit_scale
            logit = torch.mean(logit, dim=1)
            logits.append(logit)
        
        logits = torch.stack(logits)

        return logits


@TRAINER_REGISTRY.register()
class VLPL(TrainerX):
    """

    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.VLPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.VLPL.PREC == "fp32" or cfg.TRAINER.VLPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if not any(keyword in name for keyword in ["prompt_learner","prompt_argument"]):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.optim1 = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.optim2 = build_optimizer(self.model.prompt_argument, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim1, self.sched1)
        self.register_model("prompt_argument", self.model.prompt_argument, self.optim2, self.sched2)
       
        """
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        """

        self.scaler = GradScaler() if cfg.TRAINER.VLPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.VLPL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # before_params = [param.clone() for name, param in self.model.named_parameters() if "prompt_argument" in name]
            
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
            
            # after_params = [param.clone() for name, param in self.model.named_parameters() if "prompt_argument" in name]
            
            # if not torch.equal(before_params[0],after_params[0]):
            #     print("Parameter has been updated.")
            # else:
            #     print("Parameter has not been updated.")


        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
