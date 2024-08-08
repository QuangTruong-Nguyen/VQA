import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Vision(nn.Module):
    def __init__(self, vision_embedding, vision_encoder):
        super().__init__()
        self.embedding = vision_embedding
        self.encoder = vision_encoder

    def forward(self, pixel_values):
        x = self.embedding(pixel_values)
        x = self.encoder(x)
        return x

class ModalProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(768, 2048, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(2048, 2048, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class QAvision(nn.Module):
    def __init__(self,embed_llm,projector,vision, language_model, device='cpu'):

        super().__init__()
        self.embedding= embed_llm.to(device)
        
        self.projector= projector.to(device)
        self.projector.requires_grad=False
        
        self.vision_tower= vision.to(device)
        self.vision_tower.requires_grad = False
        
        self.language_model=language_model
        self.language_model.requires_grad = True
        
        self.pad_token_id = -1
        self.image_token_index=256000
        self.ignore_index=-100

    def get_embedding(self):
        return self.language_model.get_input_embedding()

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))

        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)

        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.image_token_index)


        #2
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding

        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]


        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


    def forward(self,input_ids,
                attention_mask,
                pixel_values,
                labels = None):
        inputs_embeds = self.embedding(input_ids)

        image_outputs =self.vision_tower(pixel_values)
        image_features = self.projector(image_outputs['last_hidden_state'])
        
        
        image_features=image_features.to(inputs_embeds.dtype)
    #     inputs_embeds= inputs_embeds.to(image_features.dtype)


        inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels
        )


        outputs = self.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        
        logits=outputs[0]
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )
        
        return {
            'loss':loss,
            'logits':logits,
            'embedding': inputs_embeds}