import torch
from datasets import load_dataset
from datasets import Dataset

from transformers import TrainingArguments, Trainer
from transformers.integrations import WandbCallback
# import wandb
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
from huggingface_hub import login

from data import MyDataCollator, LlavaProcessor, url_to_img
from models import QAvision, Vision, ModalProjector
from training import CustomTrainer, setup_training


#_______________________________ Initialize Wandb
# wandb.login(key='ec1748b28ac19881d87daa7a7c0d72ba6d82bd6e')
# wandb.init(project="VQA3")

#________________________________Huggingface


token = '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'
login(token)



#_______________________________ Load dataset
data = load_dataset("Vi-VLM/Vista", name="vi_llava_conversation")

data10 = data['train'][:20000]
data_val=data['validation'][:5000]

dataset10 = Dataset.from_dict(data10)
datasetVal=Dataset.from_dict(data_val)


#_______________________________ Load vision
processor_vison = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_vison = AutoModel.from_pretrained('facebook/dinov2-base')
vision=Vision(model_vison.embeddings, model_vison.encoder)

#_______________________________ Projector
projector=ModalProjector()

#_________________________________LLM
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model2 = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,   
    quantization_config=quantization_config,
)

tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
model2.resize_token_embeddings(len(tokenizer))
embed_llm=model2.get_input_embeddings()


#_________________________ Initialize processor
processor = LlavaProcessor(processor_vison, tokenizer)

# __________________________DataCollator
data_collator = MyDataCollator(processor)

#________________________ Create model components
model_qa = QAvision(embed_llm, projector, vision, model2)

#________________________ Setup training arguments
training_args = setup_training()

# Initialize trainer
trainer = CustomTrainer(
    model=model_qa,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset10,
    eval_dataset=datasetVal,
    callbacks=[WandbCallback()]
)

# Train the model
trainer.train()

torch.save(model_qa.state_dict(), 'vqa_model.pth')

# artifact = wandb.Artifact('my_model', type='model')
# artifact.add_file('vqa_model.pth')
# wandb.log_artifact(artifact)