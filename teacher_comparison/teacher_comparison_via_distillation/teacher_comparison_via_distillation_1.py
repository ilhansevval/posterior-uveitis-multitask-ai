# ============================================================================
# TEACHER COMPARISON via DISTILLATION  (downstream KD performance per teacher)
#   MacBook Pro M4 + MPS (GPU) sürümü.
#   v3 KD kodundan SAPMA YOK: aynı model, loss, sampler, scheduler, fold mantığı,
#   eşikler. Tek değişen: teacher embedding klasörü (döngü) + EMB_DIM otomatik +
#   teacher embedding L2-normalize + image_path'e göre sıralama (aynı fold).
#
#   ÇIKTI: results_teacher_compare/teacher_compare.json  (incremental) -> bana gönder.
# ============================================================================
import os, json, warnings
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # MPS'in desteklemediği op -> CPU'ya düşsün
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm   # terminal progress bar (notebook 'auto' değil -> IProgress uyarısı yok)
import timm
warnings.filterwarnings('ignore')

# ───────────────────────── CONFIG ─────────────────────────
# DATA_ROOT'u otomatik bul: teacher_embeddings_medgemma'yı içeren ilk dizin.
# Senin yapın: ~/Desktop/cerrahpasa/files/files  (script'i nereden çalıştırırsan çalıştır bulur)
_cands = [
    os.path.expanduser('~/Desktop/cerrahpasa/files/files'),
    os.path.expanduser('~/Desktop/cerrahpasa/files'),
    os.getcwd(),
    os.path.join(os.getcwd(), 'files'),
    os.path.expanduser('~/Desktop/files'),
]
DATA_ROOT = next((c for c in _cands
                  if os.path.isdir(os.path.join(c, 'teacher_embeddings_medgemma'))), None)
assert DATA_ROOT is not None, (
    "teacher_embeddings_medgemma klasörü bulunamadı.\n"
    "Denenen yollar:\n  " + "\n  ".join(_cands) +
    "\n-> Script'i 'files/files' içinde çalıştır ya da yolu elle ver."
)
print(f"DATA_ROOT = {DATA_ROOT}")
SAVE_DIR  = os.path.join(DATA_ROOT, 'results_teacher_compare')
os.makedirs(SAVE_DIR, exist_ok=True)

# (görünen ad, embedding klasörü).  Klasör yoksa otomatik atlanır.
TEACHERS = [
    ("MedGemma-4B + Ref",      "teacher_embeddings_medgemma"),
    # ("Phi-3.5-Vision + Ref", "teacher_embeddings_phi"),   # sonra
]

IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LR, NUM_FOLDS = 380, 4, 35, 1e-4, 5

# ── DEVICE: M4 -> MPS (GPU) ──
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
PIN_MEM = (DEVICE.type == 'cuda')   # pin_memory yalnız CUDA'da anlamlı; MPS'te kapalı
print(f"DEVICE = {DEVICE}  (MPS GPU kullanılıyor mu: {DEVICE.type == 'mps'})")

MAIN_LABELS = ['gt_DKS', 'gt_ODB', 'gt_VI', 'gt_MÖ']
RARE_LABELS = ['gt_DDB', 'gt_RI', 'gt_HEM', 'gt_PVK']
ALL_LABELS  = MAIN_LABELS + RARE_LABELS
MAIN_SHORT  = ['DKS', 'ODB', 'VI', 'MÖ']
RARE_SHORT  = ['DDB', 'RI', 'HEM', 'PVK']
ALL_SHORT   = MAIN_SHORT + RARE_SHORT
NUM_MAIN, NUM_RARE = 4, 4
NUM_CLASSES = 8

LAMBDA_LABEL, LAMBDA_RARE = 1.0, 2.0
LAMBDA_EMB_START, LAMBDA_EMB_END = 2.0, 0.3
MIN_THRESHOLDS = {'DDB': 0.15, 'RI': 0.15, 'HEM': 0.25, 'PVK': 0.15}
SEED = 42

# ───────────────────────── v3 components (aynen) ─────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.pos_weight=pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        p = torch.sigmoid(logits); pt = p*targets + (1-p)*(1-targets)
        at = self.alpha*targets + (1-self.alpha)*(1-targets)
        return (at*(1-pt)**self.gamma*bce).mean()

class StudentModelV3(nn.Module):
    def __init__(self, emb_dim, num_main=NUM_MAIN, num_rare=NUM_RARE):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.main_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d,512), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(512,num_main))
        self.rare_heads = nn.ModuleList([nn.Sequential(nn.Dropout(0.4), nn.Linear(d,128), nn.ReLU(),
                                       nn.Dropout(0.3), nn.Linear(128,1)) for _ in range(num_rare)])
        self.emb_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d,1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024,emb_dim))
    def forward(self,x):
        f = self.backbone(x)
        return self.main_head(f), torch.cat([h(f) for h in self.rare_heads],dim=1), F.normalize(self.emb_head(f),p=2,dim=1)

class DS(Dataset):
    def __init__(self, df, emb, tf): self.df=df.reset_index(drop=True); self.emb=emb; self.tf=tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; img=Image.open(r['image_path']).convert('RGB')
        if self.tf: img=self.tf(img)
        ml=torch.tensor([r[c] for c in MAIN_LABELS],dtype=torch.float32)
        rl=torch.tensor([r[c] for c in RARE_LABELS],dtype=torch.float32)
        return img, ml, rl, torch.tensor(self.emb[i],dtype=torch.float32)

tf_tr = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
tf_ev = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def stratified_kfold(df, n_splits=5, rs=42):
    np.random.seed(rs); n=len(df); fa=np.full(n,-1,int)
    lc=sorted([(c,int(df[c].sum())) for c in ALL_LABELS], key=lambda x:x[1])
    for c,_ in lc:
        pos=df.index[df[c]==1].values; un=pos[fa[pos]==-1]
        if len(un)==0: continue
        np.random.shuffle(un); alr=np.zeros(n_splits,int)
        for idx in pos[fa[pos]!=-1]: alr[fa[idx]]+=1
        for idx in un: t=int(np.argmin(alr)); fa[idx]=t; alr[t]+=1
    rem=np.where(fa==-1)[0]; np.random.shuffle(rem); sz=np.array([(fa==f).sum() for f in range(n_splits)])
    for idx in rem: t=int(np.argmin(sz)); fa[idx]=t; sz[t]+=1
    return [(np.where(fa!=f)[0], np.where(fa==f)[0]) for f in range(n_splits)]

def sampler_weights(df):
    mm=df[MAIN_LABELS].values; rm=df[RARE_LABELS].values
    mw=1.0/(mm.sum(0)+1); rw=1.0/(rm.sum(0)+1)*5
    s=(mm*mw).sum(1)+(rm*rw).sum(1); s=np.maximum(s,0.1); s=s/s.sum()*len(df)
    return torch.DoubleTensor(s)

def find_thresholds(yt,yp):
    th=np.zeros(NUM_CLASSES)
    for i,sn in enumerate(ALL_SHORT):
        bf,bt=0,0.5; mt=MIN_THRESHOLDS.get(sn,0.10) if sn in RARE_SHORT else 0.10
        for t in np.arange(mt,0.90,0.025):
            f1=f1_score(yt[:,i],(yp[:,i]>=t).astype(int),zero_division=0)
            if f1>bf: bf,bt=f1,t
        th[i]=bt
    return th

def remap_path(win_or_any_path):
    """Windows metadata yolunu ( ...\\<hasta>\\<dosya> ) Mac DATA_ROOT'a çevir."""
    parts = str(win_or_any_path).replace('\\', '/').split('/')
    parts = [p for p in parts if p]              # boşları at
    if len(parts) >= 2:
        return os.path.join(DATA_ROOT, parts[-2], parts[-1])   # <hasta>/<dosya>
    return os.path.join(DATA_ROOT, parts[-1])

# ───────────────────────── KD run for one teacher ─────────────────────────
def run_teacher(name, emb_dir):
    emb_path = os.path.join(DATA_ROOT, emb_dir, 'teacher_embeddings.npy')
    meta_path= os.path.join(DATA_ROOT, emb_dir, 'teacher_metadata.csv')
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        print(f"  ⏭  {name}: klasör/dosya yok ({emb_dir}) -> atlandı"); return None

    meta = pd.read_csv(meta_path, encoding='utf-8-sig')
    emb  = np.load(emb_path).astype(np.float32)
    # Windows image_path'lerini Mac'e remap et + varlık kontrolü
    meta['image_path'] = meta['image_path'].map(remap_path)
    exist = meta['image_path'].map(os.path.exists)
    print(f"  Görüntü yolu kontrolü: {int(exist.sum())}/{len(meta)} mevcut")
    if exist.sum() < len(meta):
        print(f"  ⚠️ {int((~exist).sum())} görüntü bulunamadı. Örnek beklenen yol:")
        print(f"     {meta['image_path'].iloc[0]}")
        print(f"  -> Klasör yapısı farklıysa remap_path() düzeltilmeli. İlk eksik:")
        print(f"     {meta.loc[~exist,'image_path'].iloc[0]}")
        raise FileNotFoundError("Bazı görüntüler bulunamadı — remap_path() yapısını kontrol et.")
    # tüm teacher'larda AYNI sıra -> AYNI fold
    order = np.argsort(meta['image_path'].values, kind='mergesort')
    meta  = meta.iloc[order].reset_index(drop=True); emb = emb[order]
    # paper metodu: teacher embedding L2-normalize
    emb   = emb / (np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)
    EMB_DIM = emb.shape[1]
    print(f"\n{'='*70}\n🎓 TEACHER: {name}  |  emb {emb.shape}  |  {len(meta)} görüntü\n{'='*70}")

    splits = stratified_kfold(meta, NUM_FOLDS, SEED)
    all_probs = np.zeros((len(meta),NUM_CLASSES)); all_trues = np.zeros((len(meta),NUM_CLASSES))
    fold_macros=[]

    for fold,(tr,va) in enumerate(splits):
        tdf,vdf = meta.iloc[tr], meta.iloc[va]; te,ve = emb[tr], emb[va]
        tl = DataLoader(DS(tdf,te,tf_tr), BATCH_SIZE,
                        sampler=WeightedRandomSampler(sampler_weights(tdf), len(tr)*3, replacement=True),
                        num_workers=0, pin_memory=PIN_MEM)
        vl = DataLoader(DS(vdf,ve,tf_ev), BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEM)

        model = StudentModelV3(EMB_DIM).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
        mp=tdf[MAIN_LABELS].sum().values; mpw=torch.tensor((len(tdf)-mp)/(mp+1),dtype=torch.float32).to(DEVICE)
        rp=tdf[RARE_LABELS].sum().values; rpw=torch.tensor((len(tdf)-rp)/(rp+1),dtype=torch.float32).to(DEVICE)
        mloss=FocalLoss(0.75,2.0,mpw); rloss=FocalLoss(0.85,2.5,rpw); mse=nn.MSELoss()

        print(f"  Fold {fold+1}/{NUM_FOLDS}: model hazır, eğitim başlıyor "
              f"({len(tr)} train / {len(va)} val).  İlk batch MPS derlemesi yüzünden yavaş başlar.")
        best=0; noimp=0; pat=10
        for ep in range(NUM_EPOCHS):
            model.train(); le=LAMBDA_EMB_START+(LAMBDA_EMB_END-LAMBDA_EMB_START)*(ep/NUM_EPOCHS)
            run_loss=0.0; nb=0
            pbar = tqdm(tl, desc=f"  F{fold+1} Ep{ep+1:02d}/{NUM_EPOCHS}", leave=False, ncols=90)
            for imgs,ml,rl,temb in pbar:
                imgs,ml,rl,temb = imgs.to(DEVICE),ml.to(DEVICE),rl.to(DEVICE),temb.to(DEVICE)
                mlog,rlog,pemb = model(imgs)
                loss = LAMBDA_LABEL*mloss(mlog,ml)+LAMBDA_RARE*rloss(rlog,rl)+le*mse(pemb,temb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                run_loss+=float(loss.detach()); nb+=1; pbar.set_postfix(loss=f"{run_loss/nb:.4f}")
            sch.step()
            # monitor F1 (min-threshold)
            model.eval(); P=[]; L=[]
            with torch.no_grad():
                for imgs,ml,rl,_ in vl:
                    m,r,_=model(imgs.to(DEVICE))
                    P.append(np.hstack([torch.sigmoid(m).cpu().numpy(),torch.sigmoid(r).cpu().numpy()]))
                    L.append(np.hstack([ml.numpy(),rl.numpy()]))
            P=np.vstack(P); L=np.vstack(L)
            d=np.zeros_like(P)
            for i,sn in enumerate(ALL_SHORT):
                t=MIN_THRESHOLDS.get(sn,0.3) if sn in RARE_SHORT else 0.5
                d[:,i]=(P[:,i]>=t).astype(int)
            f1m=np.mean([f1_score(L[:,i],d[:,i],zero_division=0) for i in range(NUM_CLASSES)])
            improved = f1m>best
            if improved:
                best=f1m; noimp=0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR,f'_tmp_fold{fold}.pt'))
            else:
                noimp+=1
            print(f"  F{fold+1} Ep{ep+1:02d}/{NUM_EPOCHS}  loss={run_loss/max(nb,1):.4f}  "
                  f"monitorF1m={f1m:.3f}  best={best:.3f}{'  *kaydedildi' if improved else ''}")
            if noimp>=pat:
                print(f"  -> early stop (ep {ep+1}, {pat} epoch iyileşme yok)")
                break
        print(f"  Fold {fold+1}: best monitor F1m={best:.3f}")

        # best model -> OOF probs
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR,f'_tmp_fold{fold}.pt'),map_location=DEVICE,weights_only=True)); model.eval()
        P=[]; L=[]
        with torch.no_grad():
            for imgs,ml,rl,_ in vl:
                m,r,_=model(imgs.to(DEVICE))
                P.append(np.hstack([torch.sigmoid(m).cpu().numpy(),torch.sigmoid(r).cpu().numpy()]))
                L.append(np.hstack([ml.numpy(),rl.numpy()]))
        P=np.vstack(P); L=np.vstack(L)
        all_probs[va]=P; all_trues[va]=L
        os.remove(os.path.join(SAVE_DIR,f'_tmp_fold{fold}.pt'))

    # global thresholds + metrics
    gth=find_thresholds(all_trues,all_probs); gd=np.zeros_like(all_probs)
    for i in range(NUM_CLASSES): gd[:,i]=(all_probs[:,i]>=gth[i]).astype(int)
    per_f1={}; aucs=[]
    for i,sn in enumerate(ALL_SHORT):
        per_f1[sn]=float(f1_score(all_trues[:,i],gd[:,i],zero_division=0))
        try: aucs.append(roc_auc_score(all_trues[:,i],all_probs[:,i]) if 0<all_trues[:,i].sum()<len(all_trues) else 0)
        except: aucs.append(0)
    macro=float(np.mean(list(per_f1.values())))
    rare=float(np.mean([per_f1[s] for s in RARE_SHORT]))
    main=float(np.mean([per_f1[s] for s in MAIN_SHORT]))
    res={'teacher':name,'emb_dir':emb_dir,'emb_dim':int(EMB_DIM),
         'macro_f1':round(macro,4),'main_f1':round(main,4),'rare_f1':round(rare,4),
         'macro_auc':round(float(np.mean(aucs)),4),
         'per_class_f1':{k:round(v,4) for k,v in per_f1.items()}}
    print(f"  ==> {name}: macro F1 {macro:.4f} | main {main:.4f} | rare {rare:.4f} | AUC {np.mean(aucs):.4f}")
    return res

# ───────────────────────── main loop (incremental) ─────────────────────────
out_path = os.path.join(SAVE_DIR,'teacher_compare.json')
done = {}
if os.path.exists(out_path):
    done = {r['teacher']: r for r in json.load(open(out_path,encoding='utf-8'))}
    print(f"Önceden biten: {list(done)}")

results = list(done.values())
for name, emb_dir in TEACHERS:
    if name in done:
        print(f"  ✓ {name} zaten var, atlanıyor."); continue
    r = run_teacher(name, emb_dir)
    if r is not None:
        results.append(r)
        json.dump(results, open(out_path,'w',encoding='utf-8'), indent=2, ensure_ascii=False)  # incremental save

# ───────────────────────── final table ─────────────────────────
print(f"\n{'='*70}\n📊 TEACHER COMPARISON (downstream distilled-student, KD-only)\n{'='*70}")
print(f"  {'Teacher':<24s} {'dim':>5s} {'Macro F1':>9s} {'Main F1':>8s} {'Rare F1':>8s} {'AUC':>7s}")
print(f"  {'-'*66}")
for r in sorted(results, key=lambda z:-z['macro_f1']):
    print(f"  {r['teacher']:<24s} {r['emb_dim']:>5d} {r['macro_f1']:>9.4f} {r['main_f1']:>8.4f} {r['rare_f1']:>8.4f} {r['macro_auc']:>7.4f}")
print(f"\n✅ Kaydedildi: {out_path}  (bunu bana gönder -> paper tablosunu kurarım)")
