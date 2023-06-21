### AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities

#### 1.IDEA

Alert the encoder of clip with a pretrained multilingual text encoder XLMR.

Aligned both languages and image representations by a two-stage training schema consisting of teacher learning and contrastive learning.

#### 2. Introduction

**Background**:

- Clip demonstrates impressive zero-shot performance across a number of tasks such as image classification, Image-to-Text and Textto-Image retrieval.

- Training a good language-image representation model often requires a huge amount of text-image pairs and vast computational resources.

- Existing works in the cross-lingual or multilingual setting mainly focus on the model's retrieval performance and ignores their generalization ability.

**Thought**:

- In the first stage, we use Teacher Learning to distill the knowledge learned from CLIP. 

- In the second stage, we train the model via Contrastive Learning on a relatively small amount of Chinese and English text-image pairs.

**Difference with previous work**:

- use knowledge distillation on English text pairs in addition to machine-translated text pairs;
- add human-curated translation data for better quality;
- fine-tune the model with text-image pairs to further boost its performance.

#### 3. Method

- **Teacher stage:**

  ```
  Purpose : learn a multilingual text encoder from the CLIP text encoder.
  Teacher Model : Text encoder of Clip.
  Student Model : XLM-R pretrained on multilingual data, add a mlp to transform the output dimension.
  ```

  Given parallel text input `(sent1, sent2)` 

  Teacher model encode the `sent1` get the embedding as  $x_{tos}^t$

  Student model encode the `sent2` get the embedding as $x_{cls}^s$

  Minimize the MSE between  $x_{tos}^t$ and  $x_{cls}^s$

- **Contrastive Learning stage:**

  ```
  purpose : further improve text-image alignment by contrastive learning on multilingual text-image pairs.
  image encoder : ViT, image encoder of original clip, freeze
  text encoder  : XLM-R, after Knowledge Distill.
  ```