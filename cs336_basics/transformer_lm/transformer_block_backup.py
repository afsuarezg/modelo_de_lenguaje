class my_transformer_block(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                weights: dict[str, torch.FloatTensor],
                in_features: torch.FloatTensor, 
                rope: bool=True,
                iteration: int|None=None, 
                device: torch.device=torch.device('cpu'),
                dtype: torch.dtype=torch.float32): 
   
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope=rope
        self.rope_theta = theta
        self.in_features = in_features
        self.token_positions = torch.arange(0, max_seq_len, dtype=torch.int32)
        self.device = device
        self.dtype = dtype
 
        if iteration is None:
            self.weights=weights
        else: #if type(iteration) is int:
            self.weights={}
            self.weights['ln1.weight']=weights[f'layers.{iteration}.ln1.weight']
            self.weights['ln2.weight']=weights[f'layers.{iteration}.ln2.weight']
            self.weights['attn.q_proj.weight']=weights[f'layers.{iteration}.attn.q_proj.weight']
            self.weights['attn.k_proj.weight']=weights[f'layers.{iteration}.attn.k_proj.weight']
            self.weights['attn.v_proj.weight']=weights[f'layers.{iteration}.attn.v_proj.weight']
            self.weights['attn.output_proj.weight']=weights[f'layers.{iteration}.attn.output_proj.weight']
            self.weights['ffn.w1.weight']=weights[f'layers.{iteration}.ffn.w1.weight']
            self.weights['ffn.w2.weight']=weights[f'layers.{iteration}.ffn.w2.weight']
            self.weights['ffn.w3.weight']=weights[f'layers.{iteration}.ffn.w3.weight']

        # self.RMSLayerNorm1=RMSLayerNorm(d_model=self.d_model,
        #                                eps=1e-5,
        #                                weights=self.weights['ln1.weight'],
        #                                device=self.device,
        #                                dtype=self.dtype)
        
        # self.RMSLayerNorm2=RMSLayerNorm(d_model=self.d_model,
        #                                 eps=1e-5,
        #                                 weights=self.weights['ln2.weight'],
        #                                 device=self.device,
        #                                 dtype=self.dtype)
        
        # self.causalMultiHeadSelfAttention=causalMultiHeadSelfAttention(d_model=self.d_model,
        #                                                                num_heads=self.num_heads,
        #                                                                q_proj_weight=self.weights['attn.q_proj.weight'],
        #                                                                k_proj_weight=self.weights['attn.k_proj.weight'],
        #                                                                v_proj_weight=self.weights['attn.v_proj.weight'],
        #                                                                o_proj_weight=self.weights['attn.output_proj.weight'],
        #                                                                rope=self.rope,
        #                                                                max_seq_len=self.max_seq_len,    
        #                                                                token_positions=self.token_positions,
        #                                                                theta=self.rope_theta,
        #                                                                device=self.device,
        #                                                                dtype=self.dtype)
        
        # self.positionwise_feedforward=positionwise_feedforward(d_model=self.d_model,
        #                                                         d_ff=self.d_ff,
        #                                                         w1_weight=self.weights['ffn.w1.weight'],
        #                                                         w2_weight=self.weights['ffn.w2.weight'],
        #                                                         w3_weight=self.weights['ffn.w3.weight'],
        #                                                         device=self.device,
        #                                                         dtype=self.dtype)
        

    def multihead_self_attention_sublayer(self, 
                                          in_features: torch.FloatTensor) -> torch.FloatTensor:

        x = RMSLayerNorm(d_model=self.d_model,
                         eps=1e-5,
                         weights=self.weights['ln1.weight'],
                         device=self.device,
                         dtype=self.dtype).forward(x=in_features)

        x = causalMultiHeadSelfAttention(d_model=self.d_model,
                                    num_heads=self.num_heads,
                                    q_proj_weight=self.weights['attn.q_proj.weight'],
                                    k_proj_weight=self.weights['attn.k_proj.weight'],
                                    v_proj_weight=self.weights['attn.v_proj.weight'],
                                    o_proj_weight=self.weights['attn.output_proj.weight'], 
                                    rope=self.rope, 
                                    max_seq_len=self.max_seq_len,
                                    token_positions=self.token_positions,
                                    theta=self.rope_theta).forward(x=x)

        x+=in_features
        return x


    def positionwise_feedforward_sublayer(self, 
                                          in_features: torch.FloatTensor) -> torch.FloatTensor:
        x = RMSLayerNorm(d_model=self.d_model,
                         eps=1e-5,
                         weights=self.weights['ln2.weight'],
                         device=self.device,    
                         dtype=self.dtype).forward(x=in_features)

        x = swiglu(d_model=self.d_model,
                  d_ff=self.d_ff,
                  w1_weight=self.weights['ffn.w1.weight'],
                  w2_weight=self.weights['ffn.w2.weight'],
                  w3_weight=self.weights['ffn.w3.weight'],
                  in_features=x)
        
        x+=in_features
        return x

     
    def forward(self, 
                in_features: torch.FloatTensor) -> torch.FloatTensor:
        x= self.multihead_self_attention_sublayer(in_features=in_features)

        x= self.positionwise_feedforward_sublayer(in_features=x)
        return x