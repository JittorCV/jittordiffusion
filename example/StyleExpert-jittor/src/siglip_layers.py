# copied and modified from https://github.com/bytedance/USO/blob/main/uso/flux/modules/layers.py
import torch
from torch import nn
import os 
from jittor_runtime import load_safetensors

class SigLIPMultiFeatProjModel(nn.Module):
    """
    SigLIP Multi-Feature Projection Model for processing style features from different layers 
    and projecting them into a unified hidden space.
    
    Args:
        layer_indices (list[int]): List of SigLIP hidden_states indices to extract, e.g. [-2, -11, -20]
        siglip_token_nums (int): Number of SigLIP tokens.
        style_token_nums (int): Number of style tokens.
        siglip_token_dims (int): Dimension of SigLIP tokens.
        hidden_size (int): Hidden layer size for the projection network.
        projection_layers (int): Number of linear layers in the projection network.
        context_layer_norm (bool): Whether to use context layer normalization.
        post_projection_dim (int, optional): If specified, adds a final projection layer to this dimension.
    """
    
    def __init__(
        self,
        layer_indices: list = [-2, -11, -20],
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 4096,
        projection_layers: int = 1,  # New parameter to control projection depth
        context_layer_norm: bool = True,
        post_projection_dim: int = None,
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.style_token_nums = style_token_nums
        self.hidden_size = hidden_size
        self.post_projection_dim = post_projection_dim

        # Create independent processing modules for each specified layer
        self.embedding_linears = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        activate_fn = nn.SiLU()
        
        for idx in layer_indices:
            name = str(idx)  # Use layer index as the key
            
            # Linear layer to adjust token numbers
            self.embedding_linears[name] = nn.Sequential(
                nn.Linear(siglip_token_nums, style_token_nums),
                activate_fn,
            )
            
            # Layer normalization
            self.layer_norms[name] = (
                nn.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
            )
            
            # --- MODIFIED PART ---
            proj_layers = []
            if projection_layers > 0:
                # First layer maps from siglip_token_dims to hidden_size
                proj_layers.append(nn.Linear(siglip_token_dims, hidden_size, bias=True))
                
                for _ in range(projection_layers - 1):
                    proj_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
                    proj_layers.append(activate_fn)

                if len(proj_layers) != 1: # 适配之前的架构
                    self.projections[name] = nn.Sequential(*proj_layers)
                else:
                    self.projections[name] = proj_layers[0]

            else:
                self.projections[name] = nn.Identity()

            #self.projections[name] = nn.Linear(siglip_token_dims, hidden_size, bias=True)
            # --- END OF MODIFICATION ---

        self.post_projection = None
        if self.post_projection_dim is not None:
            # Note: The input dimension here is flattened
            input_dim = hidden_size * len(layer_indices) * style_token_nums
            self.post_projection = nn.Linear(input_dim, self.post_projection_dim)

    def forward(self, siglip_outputs):
        """
        Args:
            siglip_outputs: Output from a SigLIP model, which contains the `hidden_states`.
        Returns:
            torch.Tensor: The final projected features.
        """
        first_module = next(iter(self.embedding_linears.values()))
        dtype = next(first_module.parameters()).dtype
        embeddings = []

        for idx in self.layer_indices:
            name = str(idx)
            hidden_states = siglip_outputs.hidden_states[idx]

            embedding = self._process_layer_features(
                hidden_states,
                self.embedding_linears[name],
                self.layer_norms[name],
                self.projections[name],
                dtype
            )
            embeddings.append(embedding)

        # Concatenate all embeddings along the token dimension
        embeddings = torch.cat(embeddings, dim=1)

        # If a post-projection layer is defined, apply it
        if self.post_projection is not None:
            # Flatten the tensor for the final linear layer
            bs = embeddings.shape[0]
            embeddings = embeddings.view(bs, -1) 
            embeddings = self.post_projection(embeddings)

        return embeddings

    def load_proj_model(self, checkpoint_path):
        """Loads weights for the projection model from a .pt or .safetensors checkpoint."""
        proj_model_name = 'proj_model.'
        model_state_dict = {}

        # 判断文件类型
        ext = os.path.splitext(checkpoint_path)[1].lower()

        if ext == ".safetensors":
            # 直接从 .safetensors 文件加载
            all_state_dict = load_safetensors(checkpoint_path)
        else:
            # 默认按 PyTorch checkpoint 方式加载
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            all_state_dict = checkpoint.get("model_state_dict", checkpoint)

        # 处理键名
        for k, v in all_state_dict.items():
            if k.startswith(proj_model_name):
                k = k.replace(proj_model_name, '')
            model_state_dict[k] = v

        # 加载参数
        missing, unexpected = self.load_state_dict(model_state_dict, strict=False)
        if missing or unexpected:
            print(f"[Warning] Missing keys: {missing}, Unexpected keys: {unexpected}")


    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Helper function to process features from a single layer."""
        # Adjust token numbers
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)
        
        # Normalize and project
        embedding = layer_norm(embedding)
        embedding = projection(embedding)
        return embedding
