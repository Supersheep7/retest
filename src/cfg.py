import os, yaml

def load_cfg(cfg_path="configs/default.yaml", overrides=None):
    cfg = yaml.safe_load(open(cfg_path))
    overrides = overrides or []
    
    for kv in overrides:
        k, v = kv.split("=", 1)
        cur = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            cur = cur.setdefault(kk, {})
        
        if v.lower() in {"true","false"}: v = v.lower() == "true"
        else:
            try: v = int(v)
            except: 
                try: v = float(v)
                except: ...
        cur[ks[-1]] = v
    
    def expand(x): 
        return os.path.expandvars(x) if isinstance(x, str) else x
    def walk(d):
        if isinstance(d, dict): return {k: walk(v) for k, v in d.items()}
        if isinstance(d, list): return [walk(v) for v in d]
        return expand(d)
    return walk(cfg)
