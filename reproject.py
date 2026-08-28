import clip
import torch
import torch.nn.functional as F
from utils import _prompts_for_class


@torch.no_grad()
def build_reprojection(text_proj, image_proj, drop_top=0, drop_bottom=0):
    """Construct text and image projections from the retained middle spectrum."""
    cross_operator = image_proj.T @ text_proj
    u, singular_values, vh = torch.linalg.svd(cross_operator, full_matrices=False)
    v = vh.T

    rank = singular_values.shape[0]
    if drop_top + drop_bottom >= rank:
        raise ValueError(
            f"Cannot remove {drop_top} top and {drop_bottom} bottom components from {rank}."
        )

    start = drop_top
    end = rank - drop_bottom
    u_mid = u[:, start:end]
    v_mid = v[:, start:end]

    new_text_proj = (text_proj @ v_mid @ v_mid.T).T.contiguous()
    new_image_proj = (image_proj @ u_mid @ u_mid.T).T.contiguous()
    return new_text_proj, new_image_proj


@torch.no_grad()
def encode_image_preproj_vit(clip_model, images):
    images = images.to(device=next(clip_model.parameters()).device, dtype=clip_model.dtype)
    visual = clip_model.visual

    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat(
        [
            visual.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ],
        dim=1,
    )
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    return visual.ln_post(x[:, 0, :])


@torch.no_grad()
def encode_image_preproj_rn(clip_model, images):
    """Return a ResNet visual feature before its final projection."""
    visual = clip_model.visual
    x = images.to(device=next(clip_model.parameters()).device, dtype=clip_model.dtype)

    x = visual.relu(visual.bn1(visual.conv1(x)))
    x = visual.relu(visual.bn2(visual.conv2(x)))
    x = visual.relu(visual.bn3(visual.conv3(x)))
    x = visual.avgpool(x)
    x = visual.layer1(x)
    x = visual.layer2(x)
    x = visual.layer3(x)
    x = visual.layer4(x)

    pool = visual.attnpool
    x = x.flatten(start_dim=2).permute(2, 0, 1)
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
    x = x + pool.positional_embedding[:, None, :].to(x.dtype)

    dim = x.shape[-1]
    identity = torch.eye(dim, dtype=x.dtype, device=x.device)
    zeros = torch.zeros(dim, dtype=x.dtype, device=x.device)
    bias = (
        torch.cat([pool.q_proj.bias, pool.k_proj.bias, pool.v_proj.bias])
        if pool.q_proj.bias is not None
        else None
    )

    x, _ = F.multi_head_attention_forward(
        query=x[:1],
        key=x,
        value=x,
        embed_dim_to_check=dim,
        num_heads=pool.num_heads,
        in_proj_weight=None,
        in_proj_bias=bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=zeros,
        use_separate_proj_weight=True,
        q_proj_weight=pool.q_proj.weight,
        k_proj_weight=pool.k_proj.weight,
        v_proj_weight=pool.v_proj.weight,
        training=visual.training,
        need_weights=False,
    )
    return x.squeeze(0)


@torch.no_grad()
def encode_image_preproj(clip_model, images):
    visual = clip_model.visual
    if hasattr(visual, "proj") and visual.proj is not None:
        return encode_image_preproj_vit(clip_model, images)
    if hasattr(visual, "attnpool") and hasattr(visual.attnpool, "c_proj"):
        return encode_image_preproj_rn(clip_model, images)
    raise AttributeError("Unsupported CLIP visual backbone for pre-projection encoding.")


@torch.no_grad()
def encode_text_preproj(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x[torch.arange(x.shape[0]), text.argmax(dim=-1)]


def project_text(features, text_proj):
    if features.shape[1] == text_proj.shape[0]:
        return features @ text_proj
    if features.shape[1] == text_proj.shape[1]:
        return features @ text_proj.T
    raise ValueError(
        f"Text feature dimension {features.shape[1]} does not match projection shape {tuple(text_proj.shape)}."
    )


@torch.no_grad()
def build_reproj_text_cls(classnames, template, clip_model, text_proj, prompt_bank=None):
    device = next(clip_model.parameters()).device
    text_proj = text_proj.to(device).float()
    weights = []

    for classname in classnames:
        prompts = _prompts_for_class(classname, template, prompt_bank)
        tokens = clip.tokenize(prompts).to(device)
        features = encode_text_preproj(clip_model, tokens).float()
        features = F.normalize(project_text(features, text_proj), dim=-1)
        class_feature = F.normalize(features.mean(dim=0), dim=0)
        weights.append(class_feature)

    weights = torch.stack(weights, dim=0).t().contiguous()
    return weights.half() if clip_model.dtype == torch.float16 else weights


@torch.no_grad()
def get_visual_proj(clip_model):
    """Return the visual projection with shape [output_dim, pre_feature_dim]."""
    visual = clip_model.visual
    if hasattr(visual, "proj") and visual.proj is not None:
        return visual.proj.data.float().t()
    if hasattr(visual, "attnpool") and hasattr(visual.attnpool, "c_proj"):
        return visual.attnpool.c_proj.weight.data.float()
    raise AttributeError("Unsupported CLIP visual backbone: no final projection was found.")
