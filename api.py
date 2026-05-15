
import cv2, numpy as np, os, time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch, torch.nn as nn, torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

_executor = ThreadPoolExecutor(max_workers=2)

# ── MODEL ─────────────────────────────────────────────────────────────
class MultiHeadRetinalViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0)
        self.embed_dim = self.backbone.embed_dim
        self.head_dr  = nn.Linear(self.embed_dim, 5)
        self.head_oct = nn.Linear(self.embed_dim, 4)

    def forward(self, x, task='dr'):
        features = self.backbone(x)
        return self.head_dr(features) if task == 'dr' \
               else self.head_oct(features)

# ── IMAGE VALIDATION ──────────────────────────────────────────────────
def validate_retinal_image(img_bgr: np.ndarray, modality: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason_if_invalid).

    DR  mode checks: red-dominant colour, circular dark border, square-ish ratio
    OCT mode checks: near-grayscale, horizontal layered structure, square-ish ratio
    """
    h, w = img_bgr.shape[:2]

    # ── Check 1: aspect ratio (retinal images are roughly square or wide strips) ─────
    ratio = w / h
    max_ratio = 4.5 if modality == 'oct' else 2.2
    if ratio < 0.5 or ratio > max_ratio:
        return False, (
            f"Aspect ratio {ratio:.2f} is unusual for a {modality.upper()} scan. "
            "Fundus photos are roughly square, while OCT scans can be square or wide strips."
        )

    # ── Check 2: minimum resolution ───────────────────────────────────
    if h < 64 or w < 64:
        return False, "Image resolution too low. Upload a full-resolution scan."

    # ── Compute channel stats ─────────────────────────────────────────
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_mean   = float(img_rgb[:,:,0].mean())
    g_mean   = float(img_rgb[:,:,1].mean())
    b_mean   = float(img_rgb[:,:,2].mean())

    # ── Check 3 (DR): image should be red/orange dominant ─────────────
    if modality == 'dr':
        # Fundus photos are always reddish — retinal background is red/orange
        if r_mean < g_mean or r_mean < b_mean + 5:
            return False, (
                "This does not look like a colour fundus photograph. "
                "Fundus images have a red/orange dominant background from retinal tissue. "
                "Please upload a proper retinal fundus photo."
            )

        # Fundus photos have a large circular dark border (~15-40% black pixels)
        gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < 20
        dark_pct  = float(dark_mask.mean())
        if dark_pct < 0.05:
            return False, (
                "No circular dark border detected. "
                "Standard fundus photographs have a black circular border "
                "surrounding the retinal disc. "
                "Please upload a proper fundus photograph."
            )

        # Reject face/skin-tone images: faces have mid-range uniform brightness
        # Fundus images are dark at edges and bright in centre
        # Check contrast: fundus has high local variance, faces are smooth
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 50 and dark_pct < 0.10:
            return False, (
                "Image appears to be a regular photo, not a fundus scan. "
                "Please upload a retinal fundus photograph."
            )

    # ── Check 4 (OCT): image should be near-grayscale ─────────────────
    if modality == 'oct':
        # OCT scans are greyscale (R≈G≈B)
        rg_diff = abs(r_mean - g_mean)
        rb_diff = abs(r_mean - b_mean)
        gb_diff = abs(g_mean - b_mean)
        colour_diff = (rg_diff + rb_diff + gb_diff) / 3

        if colour_diff > 25:
            return False, (
                f"OCT mode expects a grayscale scan (channel difference: {colour_diff:.0f}). "
                "The uploaded image appears to be a colour photo. "
                "Please upload a greyscale OCT cross-sectional scan."
            )

        # OCT scans are dark overall (mostly black with bright layers)
        overall_brightness = (r_mean + g_mean + b_mean) / 3
        if overall_brightness > 160:
            return False, (
                "Image is too bright for an OCT scan. "
                "OCT scans are predominantly dark with thin bright retinal layers. "
                "Please upload a proper OCT cross-sectional image."
            )

    return True, "OK"


# ── PREDICTOR ─────────────────────────────────────────────────────────
class Predictor:
    def __init__(self, model, device, modality):
        self.model    = model
        self.device   = device
        self.modality = modality
        self.model.eval()

        torch.manual_seed(42)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = True

        tile = (4, 4) if modality == 'oct' else (8, 8)

        if modality == 'dr':
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.CLAHE(clip_limit=2.0, tile_grid_size=tile, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            self.classes = ['No DR', 'Mild', 'Moderate',
                            'Severe', 'Proliferative DR']
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.CLAHE(clip_limit=2.0, tile_grid_size=tile, p=1.0),
                A.Normalize(mean=[0.5, 0.5, 0.5],
                            std =[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
            self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

        print(f"  [{modality.upper()}] Warming up...", end=" ", flush=True)
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.inference_mode():
            _ = self.model(dummy, task=modality)
        print("done ✓")

    def predict(self, img_np: np.ndarray) -> dict:
        t0 = time.perf_counter()
        tensor = self.transform(image=img_np)['image'] \
                     .unsqueeze(0).to(self.device)
        t1 = time.perf_counter()

        with torch.inference_mode():
            logits = self.model(tensor, task=self.modality)
            probs  = F.softmax(logits, dim=-1)[0].float()

        t2 = time.perf_counter()
        probs_np   = probs.cpu().numpy()
        pred_idx   = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx])

        if np.any(np.isnan(probs_np)):
            return {"error": "NaN in output — check checkpoint file."}

        print(f"  [{self.modality.upper()}] "
              f"preproc={round((t1-t0)*1000)}ms  "
              f"model={round((t2-t1)*1000)}ms  "
              f"total={round((t2-t0)*1000)}ms  "
              f"→ {self.classes[pred_idx]} ({confidence*100:.1f}%)")

        return {
            "class":         self.classes[pred_idx],
            "confidence":    confidence,
            "probabilities": {c: float(p)
                              for c, p in zip(self.classes, probs_np)},
            "modality":      self.modality,
            "inference_ms":  round((t2 - t1) * 1000)
        }

# ── STARTUP ───────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*52}")
print(f"  RetinaAI  |  Device: {device}  |  PyTorch {torch.__version__}")
print(f"{'='*52}")

model = MultiHeadRetinalViT()
ckpt  = torch.load("model/multihead_final.pth",
                   map_location=device, weights_only=False)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']
model.load_state_dict(ckpt, strict=True)
model.to(device)
model.eval()

# torch.compile() disabled on Windows (needs MSVC cl.exe)
# Uncomment below after installing Visual Studio Build Tools:
# try:
#     model = torch.compile(model, mode='reduce-overhead')
#     print("  torch.compile(): ✓")
# except Exception as e:
#     print(f"  torch.compile(): skipped — {e}")

print("  Warming up predictors:")
dr_pred  = Predictor(model, device, 'dr')
oct_pred = Predictor(model, device, 'oct')
print(f"{'='*52}")
print(f"  ✅ Ready — http://localhost:8000")
print(f"{'='*52}\n")

# ── FASTAPI ───────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Retinal Disease Detection")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    path = os.path.join("templates", "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h2>templates/index.html not found</h2>", 500)
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(
    file:     UploadFile = File(...),
    modality: str        = Form("dr")
):
    contents = await file.read()
    loop     = asyncio.get_running_loop()
    result   = await loop.run_in_executor(
        _executor, _run_predict, contents, modality, file.filename)
    return result

def _run_predict(contents: bytes, modality: str, filename: str) -> dict:
    # ── Decode ────────────────────────────────────────────────────────
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid file — could not decode image."}

    print(f"\n  Request: modality={modality}  file={filename}")

    # ── VALIDATE before inference ──────────────────────────────────────
    valid, reason = validate_retinal_image(img, modality)
    if not valid:
        print(f"  REJECTED: {reason}")
        return {
            "error":      "invalid_image",
            "message":    reason,
            "rejected":   True
        }

    # ── Run inference ─────────────────────────────────────────────────
    try:
        if modality == "dr":
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return dr_pred.predict(img_in)
        elif modality == "oct":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_in   = np.stack([img_gray] * 3, axis=-1)
            return oct_pred.predict(img_in)
        else:
            return {"error": f"Unknown modality: '{modality}'"}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"Inference error: {str(e)}"}