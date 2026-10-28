# Installs: CuPy (GPU), OpenCV, ipywidgets

!pip -q install --force-reinstall --no-cache-dir \

    cupy-cuda12x==13.0.* opencv-python-headless==4.10.* ipywidgets==8.* pandas==2.2.1



import cv2, numpy as np, cupy as cp, matplotlib.pyplot as plt, os, ipywidgets as w

from google.colab import files

from IPython.display import display, clear_output



# ── 1. Upload the map ────────────────────────────────────────────────────────

print("📤  Upload your Chennai map image (PNG/JPG)…")

uploaded = files.upload()

IMG_PATH = next(iter(uploaded))           # actual saved filename

print("✅ Using file:", IMG_PATH)



# ── 2. Convert to binary road mask ───────────────────────────────────────────

def load_map(path):

    img = cv2.imread(path)

    if img is None:

        raise ValueError(f"OpenCV can’t read {path}.")

    scale = 800 / max(img.shape[:2])

    img   = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    road  = (hsv[:,:,1] < 60) & (hsv[:,:,2] > 180)

    road  = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

    return np.where(road,255,0).astype(np.uint8), img



bin_map, rgb = load_map(IMG_PATH); H, W = bin_map.shape



# ── 3. Landmarks (lat/lon) ──────────────────────────────────────────────────

st_lat, st_lon = 13.0034, 80.2035   # St. Thomas Mount

mb_lat, mb_lon = 13.0500, 80.2820   # Marina Beach

lat0, lon0, lat1, lon1 = 12.88, 80.05, 13.22, 80.35   # rough bounds



def latlon_to_xy(lat, lon, lat0, lon0, lat1, lon1):

    x = int((lon - lon0)/(lon1-lon0) * (W-1))

    y = int((lat1 - lat)/(lat1-lat0) * (H-1))

    return x, y



# ── 4. Interactive calibration UI ───────────────────────────────────────────

def preview(lat0, lon0, lat1, lon1):

    sx, sy = latlon_to_xy(st_lat, st_lon, lat0, lon0, lat1, lon1)

    gx, gy = latlon_to_xy(mb_lat, mb_lon, lat0, lon0, lat1, lon1)

    img = rgb.copy()

    cv2.drawMarker(img,(sx,sy),(0,255,0),cv2.MARKER_CROSS,24,3)

    cv2.drawMarker(img,(gx,gy),(255,0,0),cv2.MARKER_CROSS,24,3)

    plt.figure(figsize=(6,6)); plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    plt.axis('off'); plt.title("Adjust sliders until crosses hit landmarks"); plt.show()



sl = {

  'lat0': w.FloatSlider(min=12.8,max=13.1 ,step=0.001,value=lat0,description='lat0'),

  'lon0': w.FloatSlider(min=80.0,max=80.15,step=0.001,value=lon0,description='lon0'),

  'lat1': w.FloatSlider(min=13.15,max=13.3,step=0.001,value=lat1,description='lat1'),

  'lon1': w.FloatSlider(min=80.25,max=80.4,step=0.001,value=lon1,description='lon1')

}

ui  = w.VBox(list(sl.values()))               # ← FIXED

out = w.interactive_output(preview, sl)

display(ui, out)



# ── 5. CUDA Dijkstra kernel (CuPy) ──────────────────────────────────────────

kernel = cp.RawModule(code=r'''

extern "C" _global_

void relax(float* cost,const int* pass,int W,int H,bool* changed){

 int idx = blockIdx.x*blockDim.x + threadIdx.x;

 if(idx>=W*H || !pass[idx]) return;

 int x=idx%W, y=idx/W, dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1}; float c=cost[idx];

 for(int k=0;k<4;k++){int nx=x+dx[k], ny=y+dy[k];

   if(nx>=0&&nx<W&&ny>=0&&ny<H){int n=ny*W+nx; if(!pass[n]) continue;

     float nc=c+1.f; if(nc<cost[n]){cost[n]=nc; *changed=true;}}}}''')

relax = kernel.get_function('relax')



def gpu_dijkstra(bm,start_flat):

    H,W = bm.shape

    passable = (bm==255).astype(cp.int32).ravel()

    cost = cp.full(passable.size, cp.inf, cp.float32); cost[start_flat]=0

    changed = cp.asarray([True], cp.bool_)

    bs=256; gs=(passable.size+bs-1)//bs

    while bool(changed.get()):

        changed.fill(False)

        relax((gs,), (bs,), (cost, passable, np.int32(W), np.int32(H), changed))

    return cost.reshape(H,W)



def backtrack(cost,start_rc,goal_rc):

    cost_h = cost.get()

    y,x = goal_rc; path=[(y,x)]

    dirs=[(1,0),(-1,0),(0,1),(0,-1)]

    while (y,x)!=start_rc:

        y,x = min(((y+dy,x+dx) for dx,dy in dirs

                   if 0<=y+dy<cost_h.shape[0] and 0<=x+dx<cost_h.shape[1]

                      and cost_h[y+dy,x+dx] < cost_h[y,x]),

                  key=lambda p: cost_h[p])

        path.append((y,x))

    return path[::-1]



# ── 6. Run button ───────────────────────────────────────────────────────────

def run_pipeline(_):

    clear_output(wait=True)

    display(ui, out, run_btn)

    lt0, ln0, lt1, ln1 = (sl[k].value for k in ('lat0','lon0','lat1','lon1'))

    sx, sy = latlon_to_xy(st_lat, st_lon, lt0, ln0, lt1, ln1)

    gx, gy = latlon_to_xy(mb_lat, mb_lon, lt0, ln0, lt1, ln1)



    bm = bin_map.copy()

    bm[max(0,sy-1):sy+2, max(0,sx-1):sx+2] = 255

    bm[max(0,gy-1):gy+2, max(0,gx-1):gx+2] = 255



    cost = gpu_dijkstra(cp.asarray(bm), sy*W + sx)

    if cp.isinf(cost[gy,gx]):

        print("❌  No path found—adjust bounds or use clearer map.")

        return



    path = backtrack(cost, (sy,sx), (gy,gx))

    vis = rgb.copy()

    for y,x in path: cv2.circle(vis,(x,y),1,(0,0,255),-1)

    cv2.circle(vis,(sx,sy),4,(0,255,0),-1)

    cv2.circle(vis,(gx,gy),4,(255,0,0),-1)

    plt.figure(figsize=(6,6)); plt.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))

    plt.title("GPU shortest path • St. Thomas Mount → Marina Beach")

    plt.axis('off'); plt.show()



run_btn = w.Button(description='▶ Run path‑finder', button_style='success')

run_btn.on_click(run_pipeline)

display(run_btn)
