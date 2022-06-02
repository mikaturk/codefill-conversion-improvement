import torch.nn as nn

from transformers.utils.dummy_pt_objects import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config, EncoderDecoderConfig, EncoderDecoderModel
import transformers

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    def _get_models(self):
      return self.taskmodels_dict

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained( "gpt2",
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = cls.get_encoder(model)
            else:
                setattr(model, "encoder", shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)
    

    @classmethod
    def get_encoder(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Roberta"):
            return "roberta-base"
        elif model_class_name.startswith("GPT2"):
            config = EncoderDecoderConfig.from_encoder_decoder_configs(model.config, model.config) 
            encoder_decoder = EncoderDecoderModel(config=config)
            return encoder_decoder.config.encoder
        else:
            raise KeyError(f"Add support for new model {model_class_name}")
    
    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)