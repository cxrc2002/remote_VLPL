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
from tqdm import tqdm
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
    """
    input:
    prompts's shape (k, l, d)
    tokenized_prompts (k, seq_len)

    """
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

class AttentionPool2d(nn.Module):
    """
    
    spacial_dim is patch_size   num_patch = patch_size**2
    embed_dim is channels num same as feature dim

    example: attpool = AttentionPool2d(spacial_dim=7, embed_dim=256, num_heads=8, output_dim=1024)
    before use attpool ,u need to get the patch of feature map


    input (n, dim, 7, 7)
    output(n, 1024)

    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
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

        return x[0]
    
class MutliScale_AttPooling(nn.Module):
    def __init__(self, clip_model):
        super().__init__()  
        self.dtype = clip_model.dtype

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=8, stride=8, bias=False)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=4, bias=False)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2, bias=False)
        self.scale1_attpool = AttentionPool2d(spacial_dim=7, embed_dim=256, num_heads=8, output_dim=1024)
        self.scale2_attpool = AttentionPool2d(spacial_dim=7, embed_dim=512, num_heads=8, output_dim=1024)
        self.scale3_attpool = AttentionPool2d(spacial_dim=7, embed_dim=1024, num_heads=8, output_dim=1024)
        
    def forward(self, out1, out2, out3):
        out1 = self.conv1(out1) # image patching
        out2 = self.conv2(out2)
        out3 = self.conv3(out3)
        z1 = self.scale1_attpool(out1)
        z2 = self.scale2_attpool(out2)
        z3 = self.scale3_attpool(out3) 

        return z1, z2, z3


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

    def forward(self):
        """
        out is (4, n_ctx//2, dim)

        """

        ctx = self.ctx #(4, n_ctx, dim)
        
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        prefix = self.token_prefix.unsqueeze(1).expand(-1, ctx.size(1), -1, -1)
        suffix = self.token_suffix.unsqueeze(1).expand(-1, ctx.size(1), -1, -1)

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
        # prompts (n_cls, 4, 77, dim)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.mutilsacle_attpool = MutliScale_AttPooling(clip_model).to(clip_model.dtype)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features, out1, out2, out3, out4 = self.image_encoder(image.type(self.dtype))  #(n, outdim), (n, 256, 56, 56)..
        
        out1, out2, out3 = self.mutilsacle_attpool(out1, out2, out3)  # (n, outdim)

        out1 = out1 / out1.norm(dim=-1, keepdim=True) 
        out2 = out2 / out2.norm(dim=-1, keepdim=True) 
        out3 = out3 / out3.norm(dim=-1, keepdim=True) 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        

        tokenized_prompts = self.tokenized_prompts #(n_cls, *, )
        prompts = self.prompt_learner() # prompts (n_cls, 4, *, dim)
        num_p = prompts.size(1)
        tokenized_prompts = tokenized_prompts.unsqueeze(1).expand(-1,num_p , -1)
        tokenized_prompts = tokenized_prompts.reshape((tokenized_prompts.size(0)*tokenized_prompts.size(1),tokenized_prompts.size(2)))  # (n_cls*4, *)
        prompts = prompts.reshape((prompts.size(0)*prompts.size(1), prompts.size(2), prompts.size(3)))
        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.reshape((text_features.size(0)//num_p, num_p, text_features.size(1))) # (n_cls, 4, dim)
        
        logit_scale = self.logit_scale.exp()
        logit1 = logit_scale*image_features@text_features[:,0,:].T
        logit2 = logit_scale*out1@text_features[:,1,:].T
        logit3 = logit_scale*out2@text_features[:,2,:].T
        logit4 = logit_scale*out3@text_features[:,3,:].T

        # logits = torch.stack([logit1, logit2, logit3, logit4], dim=1)
        # logits = torch.mean(logits, dim=1)
        # logits = logit1

        return logit1, logit2, logit3, logit4


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
            if not any(keyword in name for keyword in ["prompt_learner","mutilsacle_attpool"]):
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.optim1 = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.optim2 = build_optimizer(self.model.mutilsacle_attpool, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim1, self.sched1)
        self.register_model("mutilsacle_attpool", self.model.mutilsacle_attpool, self.optim2, self.sched2)
       
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
                output1, output2, output3, output4 = self.model(image)
                loss1 = F.cross_entropy(output1, label)
                loss2 = F.cross_entropy(output2, label)
                loss3 = F.cross_entropy(output3, label)
                loss4 = F.cross_entropy(output4, label)
                loss = 0.7*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # before_params = [param.clone() for name, param in self.model.named_parameters() if "prompt_argument" in name]
            
            output1, output2, output3, output4 = self.model(image)
            output = (output1 + output2 + output3 + output4) / 4 
            loss1 = F.cross_entropy(output1, label)
            loss2 = F.cross_entropy(output2, label)
            loss3 = F.cross_entropy(output3, label)
            loss4 = F.cross_entropy(output4, label)
            loss = 0.7*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4
            # loss = F.cross_entropy(output, label)
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

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        output1, output2, output3, output4 = self.model(input)
        out = 0.7*output1 + 0.1*output2 + 0.1*output3 + 0.1*output4
        return out

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label