import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from transformer_lens import HookedTransformer
import torch as t
cfg = load_cfg()

HF_TOKEN = cfg["common"]["hf_token"]

def get_model(model_name=None):

    MODEL = model_name or cfg["common"]["model"]

    mymodels = {
    'gemma': lambda: HookedTransformer.from_pretrained("gemma-2-9b", device=t.device('cpu')).half(),                                    
    'gemma_instruct': lambda: HookedTransformer.from_pretrained("gemma-2-9b-it", device=t.device('cpu')).half(),                        
    'gpt': lambda: HookedTransformer.from_pretrained("gpt2-large", device=t.device('cpu')).half(),                                      
    'gpt-j': lambda: HookedTransformer.from_pretrained("EleutherAI/gpt-j-6B", device=t.device('cpu')).half(),                           
    'llama': lambda: HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device=t.device('cpu')).half(),                       
    'llama_instruct': lambda: HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device=t.device('cpu')).half(),     
    'llama_medium': lambda: HookedTransformer.from_pretrained("Llama-2-13b", device=t.device('cpu')).half(),
    'llama_medium_instruct': lambda: HookedTransformer.from_pretrained("Llama-2-13b-chat", device=t.device('cpu')).half(),
    'pythia_deduped': lambda: HookedTransformer.from_pretrained("pythia-6.9b-deduped", device=t.device('cpu')).half(),                          
    'pythia': lambda: HookedTransformer.from_pretrained("pythia-6.9b", device=t.device('cpu')).half(),                                                 
    'yi': lambda: HookedTransformer.from_pretrained("yi-6b", device=t.device('cpu')).half(),                                            
    'yi_instruct': lambda: HookedTransformer.from_pretrained("yi-6b-chat", device=t.device('cpu')).half()                               
    }

    model = mymodels[MODEL]()
    model.to(t.device('cuda' if t.cuda.is_available() else 'cpu'))
    model.cfg.use_attn_out = False

    return model