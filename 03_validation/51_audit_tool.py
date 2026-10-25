"""Builds a self-contained HTML tool (audit/audit_tool.html) for the instance-level SAM 3 audit.

usage: python 51_audit_tool.py <seg_dir> <manifest_run.csv> <audit_dir>
- images = keys in audit_dir/user_verify_15.csv; instances numbered exactly as in 50_audit_sheets.py
  (score >= 0.6 and prompt in SHOW, file order), so the exported CSVs drop into audit_*_template.csv unchanged.
- one screen per instance (that instance highlighted on the photo) -> keys D/Y/? ; one screen per image for missed objects.
- progress autosaved in localStorage; export writes audit_instances_filled.csv and audit_images_filled.csv.
"""
import sys, io, json, base64, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_pipeline_v2"))
from imgpath import image_path

seg_dir, manifest, audit = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
SHOW = ["person", "car", "truck", "motorcycle", "chair", "table", "potted plant", "trash can", "dumpster", "utility pole",
        "street light", "utility box", "electrical cabinet"]
TR = {"person": "kişi", "car": "araba", "truck": "kamyon", "motorcycle": "motosiklet", "chair": "sandalye", "table": "masa",
      "potted plant": "saksı bitkisi", "trash can": "çöp kutusu", "dumpster": "çöp konteyneri", "utility pole": "direk",
      "street light": "sokak lambası", "utility box": "servis kutusu", "electrical cabinet": "elektrik dolabı"}
keys = pd.read_csv(audit / "user_verify_15.csv").key.tolist()
man = pd.read_csv(manifest).set_index("key")
tmpl = pd.read_csv(audit / "audit_instances_template.csv")


def jpg64(im, q=82):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def erode(m):
    e = m.copy()
    e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    return e


data = []
for key in keys:
    j = json.load(open(seg_dir / (key + ".json"), encoding="utf-8")); z = np.load(seg_dir / (key + ".npz")); shape = tuple(z["shape"])
    img = Image.open(image_path(man.loc[key])).convert("RGB")
    base = np.array(img)
    insts, idx = [], 0
    for i, d in enumerate(j["dets"]):
        if d["score"] < 0.6 or d["prompt"] not in SHOW:
            continue
        idx += 1
        m = np.unpackbits(z["masks"][i])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        out = (base * 0.35).astype(np.uint8)                        # dim everything else
        out[m] = (0.55 * base[m] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)   # tint the instance
        out[m & ~erode(m)] = (255, 255, 0)                          # yellow outline
        ys, xs = np.where(m)
        insts.append(dict(inst=idx, cls=d["prompt"], tr=TR[d["prompt"]], score=round(d["score"], 3), area=round(100 * m.mean(), 1),
                          img=jpg64(Image.fromarray(out), 80)))
    t = tmpl[tmpl.key == key]
    assert len(t) == len(insts) and list(t.inst) == [x["inst"] for x in insts] and list(t.cls) == [x["cls"] for x in insts], key
    data.append(dict(key=key, src=man.loc[key, "src"], insts=insts, sheet=jpg64(Image.open(audit / (key + ".png")), 85)))
print("images", len(data), "instances", sum(len(d["insts"]) for d in data))

HTML = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM 3 Denetim Aracı</title>
<style>
:root{--bg:#f6f6f4;--fg:#1c1c1c;--mut:#666;--card:#fff;--line:#d8d8d2;--ok:#1a7f37;--bad:#c62828;--unk:#8a6d00;--acc:#1f4fd8}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#9a9a9a;--card:#202020;--line:#3a3a3a;--ok:#5ed07a;--bad:#ff7b72;--unk:#e3b341;--acc:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif;padding:12px 16px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:6px 0 10px}.bar i{display:block;height:100%;background:var(--acc)}
.top{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline;justify-content:space-between}
h1{font-size:17px;margin:0}.mut{color:var(--mut)}
.main{display:flex;flex-wrap:wrap;gap:16px;margin-top:8px}
.imgbox{flex:1 1 560px;max-width:760px}.imgbox img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);display:block}
.side{flex:1 1 300px;max-width:440px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.cls{font-size:22px;font-weight:600}
.btns{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;padding:10px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.tp{border-color:var(--ok);color:var(--ok);font-weight:600}button.fp{border-color:var(--bad);color:var(--bad);font-weight:600}button.un{border-color:var(--unk);color:var(--unk)}
button:disabled{opacity:.4;cursor:default}
kbd{background:var(--line);border-radius:4px;padding:1px 6px;font:12px monospace}
.cur{outline:3px solid var(--acc)}
input[type=number]{width:70px;font:inherit;padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
textarea{width:100%;box-sizing:border-box;font:inherit;padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:6px 0}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;border:1px solid var(--line)}
.tp{color:var(--ok)}.fp{color:var(--bad)}.un{color:var(--unk)}
table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid var(--line);text-align:left}
</style></head><body>
<div class="top"><h1>SAM 3 denetimi <span class="mut" id="where"></span></h1><span class="mut" id="prog"></span></div>
<div class="bar"><i id="barfill" style="width:0"></i></div>
<div id="view"></div>
<script>
const DATA=__DATA__;
const LS="sam3_audit_v1";
let S={j:{},m:{},n:{}},pos=0,steps=[];
try{S=Object.assign(S,JSON.parse(localStorage.getItem(LS)||"{}"))}catch(e){}
DATA.forEach((d,di)=>{d.insts.forEach((_,ii)=>steps.push({t:"i",di,ii}));steps.push({t:"m",di})});
steps.push({t:"end"});
const jk=(d,i)=>d.key+"#"+i.inst;
const save=()=>{try{localStorage.setItem(LS,JSON.stringify(S))}catch(e){}};
// resume at first undecided step
pos=steps.findIndex(s=>s.t==="i"?!S.j[jk(DATA[s.di],DATA[s.di].insts[s.ii])]:s.t==="m"?!S.m[DATA[s.di].key]:true);if(pos<0)pos=steps.length-1;
const nInst=DATA.reduce((a,d)=>a+d.insts.length,0);
function counts(){let tp=0,fp=0,un=0;for(const k in S.j){const v=S.j[k];if(v==="TP")tp++;else if(v==="FP")fp++;else if(v==="?")un++}return{tp,fp,un}}
function render(){
  const s=steps[pos],v=document.getElementById("view"),c=counts(),done=c.tp+c.fp+c.un;
  document.getElementById("prog").textContent=`${done}/${nInst} nesne işaretlendi`;
  document.getElementById("barfill").style.width=(100*pos/(steps.length-1))+"%";
  if(s.t==="i"){
    const d=DATA[s.di],i=d.insts[s.ii],cur=S.j[jk(d,i)]||"";
    document.getElementById("where").textContent=`— görüntü ${s.di+1}/${DATA.length}: ${d.key} · nesne ${s.ii+1}/${d.insts.length}`;
    v.innerHTML=`<div class="main"><div class="imgbox"><img src="${i.img}" alt=""></div><div class="side">
    <div class="card"><div class="mut">Kırmızı + sarı çerçeveli alan, SAM 3'ün şunu bulduğu yerdir:</div>
    <div class="cls">${i.tr} <span class="mut" style="font-size:14px">(${i.cls}, skor ${i.score})</span></div></div>
    <div class="card"><b>Doğru mu?</b>
    <div class="mut" style="margin:4px 0 8px">Doğru = çerçeveli alan gerçekten bir ${i.tr} ve büyük kısmını kaplıyor.<br>Yanlış = başka bir şey, arka plan ya da sınıfı yanlış.</div>
    <div class="btns"><button class="tp ${cur==="TP"?"cur":""}" onclick="mark('TP')">Doğru <kbd>D</kbd></button>
    <button class="fp ${cur==="FP"?"cur":""}" onclick="mark('FP')">Yanlış <kbd>Y</kbd></button>
    <button class="un ${cur==="?"?"cur":""}" onclick="mark('?')">Emin değilim <kbd>?</kbd></button></div>
    <div class="btns" style="margin-top:10px"><button onclick="go(-1)" ${pos===0?"disabled":""}>← Geri <kbd>Backspace</kbd></button></div></div>
    <div class="card mut">Mevcut işaret: ${cur?`<span class="tag ${cur==="TP"?"tp":cur==="FP"?"fp":"un"}">${cur}</span>`:"yok"}<br>Yalnızca görüntüde vurgulanan nesneye bakın. Emin değilseniz "?" deyin, sonra bakarım.</div></div></div>`;
  }else if(s.t==="m"){
    const d=DATA[s.di],m=S.m[d.key]||{},nn=S.n[d.key]||"";
    document.getElementById("where").textContent=`— görüntü ${s.di+1}/${DATA.length}: ${d.key} · kaçırılanlar`;
    const f=(k,lbl,ex)=>`<div class="row"><label for="m_${k}">${lbl}<br><span class="mut" style="font-size:12px">${ex}</span></label><input id="m_${k}" type="number" min="0" step="1" value="${m[k]??""}" placeholder="0"></div>`;
    v.innerHTML=`<div class="main"><div class="imgbox"><img src="${d.sheet}" alt="">
    <div class="mut" style="margin-top:6px">Sol: orijinal. Sağ: SAM 3'ün bulduğu tüm nesneler (numaralı).</div></div><div class="side">
    <div class="card"><b>Fotoğrafta olup SAM 3'ün hiç işaretlemediği nesneler</b>
    <div class="mut" style="margin:4px 0 8px">Sağdaki resimde maskesi olmayan, net görünen nesneleri sayın. Çok uzaktaki/çok küçük olanları saymayın. Yoksa 0 yazın (ya da boş bırakın).</div>
    ${f("missed_vehicle","Araç","araba, kamyon, motosiklet")}${f("missed_furniture","Mobilya/bitki","sandalye, masa, saksı bitkisi")}
    ${f("missed_infrastructure","Altyapı","çöp kutusu, konteyner, direk, sokak lambası, servis kutusu")}${f("missed_person","Kişi","")}
    <div class="row" style="display:block"><label for="note">Not (isteğe bağlı)</label><textarea id="note" rows="2">${nn}</textarea></div>
    <div class="btns" style="margin-top:8px"><button onclick="go(-1)">← Geri</button><button class="tp" onclick="saveM();go(1)">Kaydet ve devam <kbd>Enter</kbd></button></div></div></div></div>`;
  }else{
    document.getElementById("where").textContent="— bitti";
    const P=(a,b)=>b?(100*a/b).toFixed(1)+" %":"–";
    const byC={};DATA.forEach(d=>d.insts.forEach(i=>{const r=S.j[jk(d,i)];if(!r)return;(byC[i.cls]??={tp:0,fp:0,un:0});byC[i.cls][r==="TP"?"tp":r==="FP"?"fp":"un"]++}));
    const rows=Object.entries(byC).map(([k,x])=>`<tr><td>${k}</td><td>${x.tp}</td><td>${x.fp}</td><td>${x.un}</td><td>${P(x.tp,x.tp+x.fp)}</td></tr>`).join("");
    v.innerHTML=`<div class="card"><h2 style="margin-top:0">Tamamlandı</h2>
    <p>Toplam ${c.tp+c.fp+c.un}/${nInst} nesne: <b class="tp">${c.tp} doğru</b>, <b class="fp">${c.fp} yanlış</b>, <b class="un">${c.un} emin değil</b>. Kesinlik (precision) = <b>${P(c.tp,c.tp+c.fp)}</b>.</p>
    <table><tr><th>sınıf</th><th>TP</th><th>FP</th><th>?</th><th>kesinlik</th></tr>${rows}</table>
    <div class="btns" style="margin-top:14px"><button class="tp" onclick="exp()">CSV dosyalarını indir</button><button onclick="go(-1)">← Geri</button></div>
    <p id="dlmsg" class="mut"></p><details style="margin:8px 0"><summary>İndirme çalışmazsa: metni kopyalayın</summary>
    <p class="mut">Her kutuyu tıklayıp Ctrl+A, Ctrl+C ile kopyalayın ve bana sohbete yapıştırın (önce nesneler, sonra görüntüler).</p>
    <div><b>1) Nesneler</b> <button onclick="selAll('t1')">Tümünü seç</button><br><textarea id="t1" rows="6" readonly>${build().I}</textarea></div>
    <div style="margin-top:8px"><b>2) Görüntüler</b> <button onclick="selAll('t2')">Tümünü seç</button><br><textarea id="t2" rows="6" readonly>${build().M}</textarea></div></details>
    <p class="mut">İki dosya iner: <code>audit_instances_filled.csv</code> ve <code>audit_images_filled.csv</code> (İndirilenler klasörü). Sonra bana "bitti" yazmanız yeterli.<br>
    İşaretlemeler bu tarayıcıda otomatik saklanır; sekmeyi kapatıp sonra devam edebilirsiniz (aynı tarayıcı ve aynı dosya).</p></div>`;
  }
}
function mark(v){const s=steps[pos],d=DATA[s.di];S.j[jk(d,d.insts[s.ii])]=v;save();go(1)}
function saveM(){const s=steps[pos],d=DATA[s.di],o={};["missed_vehicle","missed_furniture","missed_infrastructure","missed_person"].forEach(k=>{const e=document.getElementById("m_"+k);o[k]=e.value===""?0:Math.max(0,parseInt(e.value)||0)});S.m[d.key]=o;S.n[d.key]=document.getElementById("note").value;save()}
function go(n){pos=Math.min(steps.length-1,Math.max(0,pos+n));render();window.scrollTo(0,0)}
function csv(rows){return "﻿"+rows.map(r=>r.map(x=>{x=String(x??"");return /[",\n]/.test(x)?'"'+x.replace(/"/g,'""')+'"':x}).join(",")).join("\r\n")+"\r\n"}
function dl(name,text){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type:"text/csv;charset=utf-8"}));a.download=name;document.body.appendChild(a);a.click();a.remove()}
function build(){
  const I=[["key","inst","cls","score","judgement"]],M=[["key","src","n_inst","missed_vehicle","missed_furniture","missed_infrastructure","missed_person","note"]];
  DATA.forEach(d=>{d.insts.forEach(i=>I.push([d.key,i.inst,i.cls,i.score,S.j[jk(d,i)]||""]));const m=S.m[d.key]||{};M.push([d.key,d.src,d.insts.length,m.missed_vehicle??"",m.missed_furniture??"",m.missed_infrastructure??"",m.missed_person??"",S.n[d.key]||""])});
  return{I:csv(I),M:csv(M)}}
function exp(){const b=build();dl("audit_instances_filled.csv",b.I);setTimeout(()=>dl("audit_images_filled.csv",b.M),400);document.getElementById("dlmsg").textContent="İndirme başlatıldı. Dosyalar İndirilenler klasöründe görünmüyorsa aşağıdaki metin kutularını kullanın."}
function selAll(id){const e=document.getElementById(id);e.focus();e.select()}
document.addEventListener("keydown",e=>{
  if(e.target.matches("input,textarea")){if(e.key==="Enter"&&steps[pos].t==="m"){e.preventDefault();saveM();go(1)}return}
  const s=steps[pos];
  if(s.t==="i"){const k=e.key.toLowerCase();if(k==="d"||k==="1")mark("TP");else if(k==="y"||k==="2")mark("FP");else if(k==="?"||k==="e"||k==="3")mark("?");else if(e.key==="Backspace")go(-1)}
  else if(s.t==="m"&&e.key==="Enter"){saveM();go(1)}
  else if(e.key==="Backspace")go(-1)});
render();
</script></body></html>"""
(audit / "audit_tool.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
print("wrote", audit / "audit_tool.html", round((audit / "audit_tool.html").stat().st_size / 1e6, 1), "MB")
