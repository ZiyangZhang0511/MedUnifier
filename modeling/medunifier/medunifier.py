from tqdm.auto import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torchvision.utils import save_image

from lavis.common.registry import registry
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from modeling.medunifier.tools import all_gather_with_grad, concat_all_gather
from modeling.medunifier.blip2 import (
    Blip2Base,
    disabled_train,
)

from modeling.vae import vqvae
from utilities.metrics import compute_recall_at_k, compute_precK, compute_image_captioning_metric



class MedUnifier(Blip2Base):

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        latent_dim=768,
        codebook_size=512,

        requires_ITC=True,
        requires_ITM=True,
        requires_ITG=True,
        requires_TIG=True,
    ):
        super().__init__()

        self.requires_ITC = requires_ITC
        self.requires_ITM = requires_ITM
        self.requires_ITG = requires_ITG
        self.requires_TIG = requires_TIG

        self.vit_model = vit_model

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        if self.requires_ITC:
            self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        if self.requires_ITM:
            self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len

        if self.requires_TIG:
            self.vis_dim = self.visual_encoder.embed_dim # local visual embedding dimensions
            txt_dim = self.Qformer.config.hidden_size # text embedding dimension
            self.vqvae = vqvae.VQVAE(self.vis_dim, txt_dim, latent_dim, codebook_size, mode=self.mode)


    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]
        
        image_embeds = self.ln_vision(self.visual_encoder(image))
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        image_feats_all = image_feats
        text_feat_all = text_feat

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
       
        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = 0

        bs = image.size(0)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )
                   
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        image_embeds_world = image_embeds

        with torch.no_grad():  
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            if torch.any(torch.isnan(weights_t2i[b])) or torch.any(torch.isinf(weights_t2i[b])) or torch.any(weights_t2i[b] < 0):
                print(weights_t2i[b], sim_t2i, sim_i2t, loss_itc)
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)

        if self.requires_ITM:
            loss_itm = F.cross_entropy(logits, itm_labels)
        else:
            loss_itm = 0.


        ###============== Text-grounded Image Generation =============###
        local_visual_embeddings = image_embeds[:, 1:, :]
        text_embeddings = text_output.last_hidden_state[:, 0, :].unsqueeze(dim=1)
        multimodal_embeddings = query_output.last_hidden_state
        
        output_vae = self.vqvae(local_visual_embeddings, text_embeddings, multimodal_embeddings)   
        image_recon = output_vae[0]
        loss_codebook = output_vae[1]
        perplexity_top = output_vae[2]
        perplexity_bottom = output_vae[3]

        if self.requires_TIG:
            loss_recon = F.mse_loss(image_recon, image)
            loss_tig = loss_codebook + loss_recon
        else:
            loss_tig = 0.


        ##================= Imgae-grounded Text Generation ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        if self.requires_ITG:
            loss_itg = lm_output.loss
        else:
            loss_itg = 0.

        return {
            "loss": loss_itc + loss_itm + loss_itg + loss_tig,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_itg": loss_itg,
            "loss_tig": loss_tig,
            # "image_recon": image_recon.detach().cpu(),
        }

    @torch.no_grad()
    def get_encoding_ids(self, samples):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
        Returns:
            id_top (Tensor): encoding code at top level
            id_bottom (Tensor): encoding code at bottom level
        """

        image = samples["image"]
        text = samples["text_input"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        local_visual_embeddings = image_embeds[:, 1:, :]
        text_embeddings = text_output.last_hidden_state[:, 0, :].unsqueeze(dim=1)
        multimodal_embeddings = query_output.last_hidden_state

        id_top, id_bottom = self.vqvae.get_ids(local_visual_embeddings, text_embeddings, multimodal_embeddings)

        return id_top, id_bottom

    @torch.no_grad()
    def generate_images(self, code_top, code_bottom):
        generated_images = self.vqvae.decode_code(code_top, code_bottom)
        return generated_images


    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        cur_step=0,
        requires_rk=False,
        requires_pk=False,
        requires_cp=False,
        requires_itm=False,
        k_test=256,
    ):
        """
        Function: evaluate model during pre-training
        """
        self.training = False
        metrics_dict = {}

        epoch_loss = 0
        epoch_loss_tig = 0
        epoch_loss_itc = 0
        epoch_loss_itm = 0
        epoch_loss_itg = 0
        for batch in dataloader:
            batch["image"] = batch["image"].to(self.device)

            output = self.forward(batch)
            
            loss = output["loss"]
            loss_vae = output["loss_tig"]
            loss_itc = output["loss_itc"]
            loss_itm = output["loss_itm"]
            loss_lm = output["loss_itg"]

            epoch_loss += loss.item() / len(dataloader)
            epoch_loss_tig += loss_tig.item() / len(dataloader)
            epoch_loss_itc += loss_itc.item() / len(dataloader)
            epoch_loss_itm += loss_itm.item() / len(dataloader)
            epoch_loss_itg += loss_itg.item() / len(dataloader)

        metrics_dict["loss"] = epoch_loss
        metrics_dict["loss_vae"] = epoch_loss_vae
        metrics_dict["loss_itc"] = epoch_loss_itc
        metrics_dict["loss_itm"] = epoch_loss_itm
        metrics_dict["loss_lm"] = epoch_loss_lm

        if requires_rk or requires_pk:
            score_sim_i2t, score_sim_t2i = self.compute_score_matrix(dataloader, k_test=k_test, requires_itm=requires_itm)

        if requires_rk:
            r1_i2t = compute_recall_at_k(score_sim_i2t, k=1)
            r5_i2t = compute_recall_at_k(score_sim_i2t, k=5)
            r10_i2t = compute_recall_at_k(score_sim_i2t, k=10)

            r1_t2i = compute_recall_at_k(score_sim_t2i, k=1)
            r5_t2i = compute_recall_at_k(score_sim_t2i, k=5)
            r10_t2i = compute_recall_at_k(score_sim_t2i, k=10)

            metrics_dict["r1_i2t"] = r1_i2t
            metrics_dict["r5_i2t"] = r5_i2t
            metrics_dict["r10_i2t"] = r10_i2t
            metrics_dict["r1_t2i"] = r1_t2i
            metrics_dict["r5_t2i"] = r5_t2i
            metrics_dict["r10_t2i"] = r10_t2i

        if requires_pk:
            p1_i2t = compute_precK(score_sim_i2t, dataloader.dataset.img2txt, k=1)
            p5_i2t = compute_precK(score_sim_i2t, dataloader.dataset.img2txt, k=5)
            p10_i2t = compute_precK(score_sim_i2t, dataloader.dataset.img2txt, k=10)

            p1_t2i = compute_precK(score_sim_t2i, dataloader.dataset.txt2img, k=1)
            p5_t2i = compute_precK(score_sim_t2i, dataloader.dataset.txt2img, k=5)
            p10_t2i = compute_precK(score_sim_t2i, dataloader.dataset.txt2img, k=10)

            metrics_dict["p1_i2t"] = p1_i2t
            metrics_dict["p5_i2t"] = p5_i2t
            metrics_dict["p10_i2t"] = p10_i2t
            metrics_dict["p1_t2i"] = p1_t2i
            metrics_dict["p5_t2i"] = p5_t2i
            metrics_dict["p10_t2i"] = p10_t2i

        if requires_cp:
            pred_captions_all = []
            gt_captions_all = dataloader.dataset.ip_gts
            for batch in tqdm(dataloader):
                batch["image"] = batch["image"].to(self.device)
                pred_captions = self.generate(
                    batch,
                    max_length=self.max_txt_len,
                    min_length=3,
                )
                pred_captions_all.extend(pred_captions)
            
            cp_dict = compute_image_captioning_metric(pred_captions_all, gt_captions_all)

            for key, value in cp_dict.items():
                metrics_dict[key] = value

        return metrics_dict

    @torch.no_grad()
    def compute_score_matrix(self, dataloader, k_test=256, requires_itm=False):
        """
        Function: return a similarity score matrix on dataloader
        """
        ###======== text embeddings ========###
        texts = dataloader.dataset.texts
        num_texts = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []

        for i in range(0, num_texts, text_bs):
            text = texts[i : min(num_texts, i + text_bs)]
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            text_feat = self.forward_text(text_input)
            text_embed = F.normalize(self.text_proj(text_feat))
            text_id = text_input.input_ids
            text_att = text_input.attention_mask
            
            text_embeds.append(text_embed)
            text_ids.append(text_id)
            text_atts.append(text_att)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)

        ###======== image embeddings ========###
        vit_feats = []
        image_embeds = []

        for samples in dataloader:
            image = samples["image"]

            image = image.to(self.device)
            image_feat, vit_feat = self.forward_image(image)
            image_embed = self.vision_proj(image_feat)
            image_embed = F.normalize(image_embed, dim=-1)

            vit_feats.append(vit_feat.cpu())
            image_embeds.append(image_embed)

        vit_feats = torch.cat(vit_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)

        ###======== i2t score ======== ###
        sims_matrix = []
        for image_embed in image_embeds:
            sim_q2t = image_embed @ text_embeds.t()
            sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)

        score_matrix_i2t = torch.full(
            (len(dataloader.dataset.image_relpaths), len(texts)), -100.0
        ).to(self.device)


        for i, sims in enumerate(tqdm(sims_matrix)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

            if requires_itm:
                image_inputs = vit_feats[i].repeat(k_test, 1, 1).to(self.device)
                score = self.compute_itm(
                    image_inputs=image_inputs,
                    text_ids=text_ids[topk_idx],
                    text_atts=text_atts[topk_idx],
                ).float()
            else:
                score = 0.

            score_matrix_i2t[i, topk_idx] = score + topk_sim

        ###======== t2i score ========###
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(texts), len(dataloader.dataset.image_relpaths)), -100.0
        ).to(self.device)
        for i, sims in enumerate(tqdm(sims_matrix)):
            topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
            
            if requires_itm:
                image_inputs = vit_feats[topk_idx.cpu()].to(self.device)
                score = self.compute_itm(
                    image_inputs=image_inputs,
                    text_ids=text_ids[i].repeat(k_test, 1),
                    text_atts=text_atts[i].repeat(k_test, 1),
                ).float()
            else:
                score = 0.

            score_matrix_t2i[i, topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


    def forward_image(self, image):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))

            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))

            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )
