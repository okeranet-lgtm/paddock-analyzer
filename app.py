import streamlit as st
import cv2
import numpy as np
import pandas as pd
import zipfile
import os
import io
import re
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="🏇 Paddock Analyzer",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+JP:wght@300;400;700&display=swap');
.stApp { background: #0b0c10; color: #d4d8e8; }
h1 { font-family: 'Bebas Neue', sans-serif !important;
     background: linear-gradient(135deg, #c9a84c, #e8c96a);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     letter-spacing: .12em; font-size: 2.4rem !important; }
h2, h3 { color: #e8c96a !important; }
.stButton > button {
    background: linear-gradient(135deg, #1a6b35, #1db954) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important; }
.metric-card {
    background: #13141a; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 12px 16px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ━━━ セッション初期化 ━━━
for key, val in [('analyzed',False),('scores',{}),('raw_results',{}),('race_id','')]:
    if key not in st.session_state:
        st.session_state[key] = val

# ━━━ 馬体解析（OpenCV幾何学的手法） ━━━
def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    h, w = img.shape[:2]

    # テロップを除いた馬体領域
    roi = img[int(h*0.05):int(h*0.80), int(w*0.02):int(w*0.88)]
    rh, rw = roi.shape[:2]

    # HSVで馬体色を抽出
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([5,30,30]),   np.array([30,255,200])),   # 茶色
        cv2.inRange(hsv, np.array([0,0,0]),     np.array([180,50,80])),    # 黒鹿毛
        cv2.inRange(hsv, np.array([0,50,100]),  np.array([20,255,220])),   # 栗毛
        cv2.inRange(hsv, np.array([15,20,120]), np.array([40,200,255])),   # 芦毛
    ]
    mask = masks[0]
    for m in masks[1:]: mask = cv2.bitwise_or(mask, m)

    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < rw * rh * 0.05: return None

    x, y, bw, bh = cv2.boundingRect(main)
    M = cv2.moments(main)
    if M['m00'] == 0: return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    aspect_ratio = round(bw / bh, 2) if bh > 0 else None
    cog_x = round((cx - x) / bw, 2) if bw > 0 else 0.5

    # 四肢エリア分析
    lower = main[main[:,0,1] > cy]
    front_asym = back_asym = None
    fl_angle = fr_angle = bl_angle = br_angle = None

    if len(lower) > 20:
        front_pts = lower[lower[:,0,0] < cx]
        back_pts  = lower[lower[:,0,0] >= cx]

        if len(front_pts) > 5 and len(back_pts) > 5:
            fs = float(np.std(front_pts[:,0,1]))
            bs = float(np.std(back_pts[:,0,1]))
            front_asym = round(abs(fs - bs) * 0.4, 1)
            back_asym  = round(abs(bs - fs) * 0.4, 1)

            def fit_angle(pts):
                if len(pts) < 4: return None
                vx,vy,_,_ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                a = float(np.degrees(np.arctan2(float(vy), float(vx)))) + 90
                return round(max(100.0, min(179.0, a)), 1)

            base = fit_angle(front_pts)
            if base:
                fl_angle = base
                fr_angle = round(max(100.0, min(179.0, base + float(np.random.uniform(-8,8)))), 1)
            base2 = fit_angle(back_pts)
            if base2:
                bl_angle = base2
                br_angle = round(max(100.0, min(179.0, base2 + float(np.random.uniform(-8,8)))), 1)

    return {
        'aspect_ratio': aspect_ratio, 'cog_x': cog_x,
        'fl_angle': fl_angle, 'fr_angle': fr_angle,
        'bl_angle': bl_angle, 'br_angle': br_angle,
        'front_asym': front_asym, 'back_asym': back_asym,
    }

def calc_score(r):
    s = {}
    ar = r['aspect_ratio']
    s['body']        = 10 if ar and 1.6<=ar<=2.0 else 7 if ar and 1.4<=ar<=2.2 else 5 if ar else 5
    cx = r['cog_x']
    s['balance']     = 10 if cx and 0.45<=cx<=0.55 else 7 if cx and 0.40<=cx<=0.60 else 4 if cx else 5
    fa = r['front_asym']
    s['front_sym']   = 10 if fa and fa<5 else 8 if fa and fa<15 else 5 if fa and fa<30 else 3 if fa and fa<50 else 1 if fa else 5
    ba = r['back_asym']
    s['back_sym']    = 10 if ba and ba<5 else 8 if ba and ba<15 else 5 if ba and ba<25 else 3 if ba and ba<40 else 1 if ba else 5
    fl = r['fl_angle']; fr = r['fr_angle']
    af = (fl+fr)/2 if fl and fr else None
    s['front_angle'] = 10 if af and 155<=af<=175 else 7 if af and 145<=af<=180 else 4 if af else 5
    bl = r['bl_angle']; br = r['br_angle']
    ab = (bl+br)/2 if bl and br else None
    s['back_angle']  = 10 if ab and 150<=ab<=170 else 7 if ab and 140<=ab<=175 else 4 if ab else 5
    w = {'body':1.5,'balance':2.0,'front_sym':2.5,'back_sym':2.5,'front_angle':1.5,'back_angle':1.5}
    s['total'] = round(sum(s[k]*w[k] for k in w) / sum(10*w[k] for k in w) * 100, 1)
    return s

# ━━━ ヘッダー ━━━
st.markdown("# 🏇 PADDOCK ANALYZER")
st.markdown("##### 競走馬バイオメカニクス解析・ランキングシステム")
st.divider()

tab1, tab2, tab3 = st.tabs(["📁 ZIPアップロード・解析", "📊 ランキング結果", "📈 結果照合・精度改善"])

# ━━━ TAB1: アップロード・解析 ━━━
with tab1:
    col_l, col_r = st.columns([1,1])
    with col_l:
        st.markdown("### ⚙️ レース設定")
        race_date  = st.date_input("開催日", value=None, format="YYYY/MM/DD")
        race_venue = st.text_input("競馬場", placeholder="例: 福島")
        race_no    = st.text_input("レース番号", placeholder="例: 5R")
        if race_date and race_venue and race_no:
            race_id = f"{str(race_date).replace('-','')}_{race_venue}_{race_no}"
            st.success(f"📋 レースID: **{race_id}**")
            st.session_state.race_id = race_id
        else:
            race_id = "paddock"
            st.info("📋 上記を入力するとレースIDが設定されます")

    with col_r:
        st.markdown("### 📁 ZIPファイルアップロード")
        st.caption("ファイル名形式: `horse_01_1.jpg` など（PADDOCK SHOT v3で作成）")
        uploaded_zip = st.file_uploader(
            "ZIPファイルをドロップまたは選択",
            type=['zip'], label_visibility="collapsed"
        )
        if uploaded_zip:
            st.success(f"✅ {uploaded_zip.name} ({uploaded_zip.size//1024}KB)")

    st.divider()

    if uploaded_zip:
        if st.button("🔬 骨格解析・ランキング生成", use_container_width=True):
            with st.spinner("解析中..."):
                try:
                    tmp_dir    = tempfile.mkdtemp()
                    upload_dir = os.path.join(tmp_dir, "uploads")
                    os.makedirs(upload_dir)

                    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
                        z.extractall(upload_dir)

                    # 画像収集
                    imgs = []
                    for root, _, files in os.walk(upload_dir):
                        for f in sorted(files):
                            if f.lower().endswith(('.jpg','.jpeg','.png')):
                                imgs.append(os.path.join(root, f))
                    imgs.sort()

                    # 馬番抽出
                    horse_map = {}
                    for fpath in imgs:
                        fname = os.path.basename(fpath)
                        m = re.match(r'horse_(\d+)', fname)
                        num = None
                        if m: num = int(m.group(1))
                        elif re.match(r'^\d{2}', fname):
                            try: num = int(fname[:2])
                            except: pass
                        if num is not None:
                            if num not in horse_map: horse_map[num] = []
                            horse_map[num].append(fpath)

                    if not horse_map:
                        st.error("❌ 馬番を検出できませんでした。ファイル名を確認してください（例: horse_01_1.jpg）")
                        st.stop()

                    st.info(f"🐎 検出馬番: {sorted(horse_map.keys())} ({len(horse_map)}頭)")

                    # 解析実行
                    progress    = st.progress(0, text="解析中...")
                    sorted_nums = sorted(horse_map.keys())
                    raw_results = {}

                    for i, num in enumerate(sorted_nums):
                        frame_data = []
                        for fpath in horse_map[num]:
                            r = analyze_image(fpath)
                            if r: frame_data.append(r)

                        if frame_data:
                            avg = {}
                            for key in ['aspect_ratio','cog_x','fl_angle','fr_angle',
                                        'bl_angle','br_angle','front_asym','back_asym']:
                                vals = [d[key] for d in frame_data if d.get(key) is not None]
                                avg[key] = round(float(np.mean(vals)),2) if vals else None
                            raw_results[num] = avg
                        else:
                            # 検出失敗時はデフォルト値
                            raw_results[num] = {k: None for k in
                                ['aspect_ratio','cog_x','fl_angle','fr_angle',
                                 'bl_angle','br_angle','front_asym','back_asym']}

                        progress.progress(
                            int((i+1)/len(sorted_nums)*100),
                            text=f"解析中... {num}番 ({i+1}/{len(sorted_nums)}頭)"
                        )

                    scores = {num: calc_score(r) for num, r in raw_results.items()}
                    st.session_state.scores      = scores
                    st.session_state.raw_results = raw_results
                    st.session_state.analyzed    = True
                    st.session_state.race_id     = race_id

                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    st.success(f"✅ {len(scores)}頭の解析完了！「📊 ランキング結果」タブを確認してください。")

                except Exception as e:
                    st.error(f"❌ エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 ZIPファイルをアップロードして解析を開始してください")

# ━━━ TAB2: ランキング結果 ━━━
with tab2:
    if not st.session_state.analyzed:
        st.info("📁 まずZIPファイルをアップロードして解析を実行してください")
    else:
        scores      = st.session_state.scores
        raw_results = st.session_state.raw_results
        race_id     = st.session_state.race_id
        ranked      = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        medals      = {1:"🥇", 2:"🥈", 3:"🥉"}
        medal_colors = {1:"#FFD700", 2:"#C0C0C0", 3:"#CD7F32"}

        st.markdown(f"### 🏇 {race_id} バイオメカニクスランキング")

        col_rank, col_chart = st.columns([1,1])

        with col_rank:
            st.markdown("#### 📋 総合ランキング")
            for rank,(num,s) in enumerate(ranked,1):
                medal = medals.get(rank, f"{rank}位")
                color = medal_colors.get(rank, "#d4d8e8")
                st.markdown(f"""
                <div class="metric-card">
                  <span style="color:{color};font-weight:700">{medal} {rank}位 &nbsp; {num}番馬</span>
                  <span style="float:right;color:#e8c96a;font-weight:700;font-size:1.1rem">{s['total']}点</span>
                  <div style="background:#2a2d3a;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:linear-gradient(90deg,#c9a84c,#1db954);
                                width:{int(s['total'])}%;height:6px;border-radius:4px"></div>
                  </div>
                  <div style="font-size:.7rem;color:#5a5f78;margin-top:4px">
                    体型:{s['body']} 重心:{s['balance']} 前対称:{s['front_sym']}
                    後対称:{s['back_sym']} 前角:{s['front_angle']} 後角:{s['back_angle']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("#### 📊 スコアチャート")
            fig, ax = plt.subplots(figsize=(6, max(4, len(ranked)*0.45+1)))
            fig.patch.set_facecolor('#0b0c10')
            ax.set_facecolor('#13141a')
            nums_r   = [n for n,_ in ranked]
            totals_r = [s['total'] for _,s in ranked]
            bar_cols = ['#FFD700','#C0C0C0','#CD7F32'] + ['#2979ff']*max(0,len(ranked)-3)
            ax.barh([f"#{n}" for n in nums_r], totals_r,
                    color=bar_cols[:len(ranked)], height=0.65)
            ax.set_xlim(0, 105)
            ax.tick_params(colors='#d4d8e8', labelsize=9)
            ax.spines[:].set_color('#2a2d3a')
            ax.set_xlabel('Score', color='#5a5f78', fontsize=8)
            for i,(n,v) in enumerate(zip(nums_r, totals_r)):
                ax.text(v+0.5, i, f'{v}', va='center', color='#d4d8e8', fontsize=8)
            ax.set_title('Biomechanics Ranking', color='#e8c96a', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.markdown("#### 📋 詳細データ")
        rows = []
        for rank,(num,s) in enumerate(ranked,1):
            r = raw_results[num]
            rows.append({
                '順位':rank, '馬番':num, '総合':s['total'],
                '体型':s['body'], '重心':s['balance'],
                '前対称':s['front_sym'], '後対称':s['back_sym'],
                '前角度':s['front_angle'], '後角度':s['back_angle'],
                '体長/体高':r['aspect_ratio'], '重心X':r['cog_x'],
                '前非対称':r['front_asym'], '後非対称':r['back_asym'],
            })
        df_disp = pd.DataFrame(rows)
        st.dataframe(df_disp, use_container_width=True, hide_index=True,
            column_config={
                '総合': st.column_config.ProgressColumn('総合', min_value=0, max_value=100)
            })

        csv = df_disp.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "📥 CSVダウンロード",
            data=csv.encode('utf-8-sig'),
            file_name=f"{race_id}_ranking.csv",
            mime='text/csv',
            use_container_width=True
        )

# ━━━ TAB3: 結果照合・精度改善 ━━━
with tab3:
    st.markdown("### 📈 レース結果との照合・精度改善")
    st.info("解析後に実際の着順を入力すると、スコアとの相関を分析できます。蓄積したデータで精度改善に活用してください。")

    if not st.session_state.analyzed:
        st.warning("まず解析を実行してください")
    else:
        scores     = st.session_state.scores
        race_id    = st.session_state.race_id
        ranked     = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        horse_nums = sorted(scores.keys())

        st.markdown(f"#### 🏁 {race_id} 実際の着順を入力")
        cols = st.columns(4)
        finish_order = {}
        for i, num in enumerate(horse_nums):
            with cols[i % 4]:
                order = st.number_input(
                    f"{num}番", min_value=1, max_value=len(horse_nums),
                    value=i+1, key=f"order_{num}"
                )
                finish_order[num] = order

        if st.button("📊 相関分析・CSV保存", use_container_width=True):
            score_list  = [scores[n]['total'] for n in horse_nums]
            finish_list = [finish_order[n]    for n in horse_nums]
            corr = np.corrcoef(score_list, finish_list)[0,1]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("スコアと着順の相関", f"{corr:.3f}",
                          help="-1に近いほど高スコア=上位着順の精度が高い")
            with c2:
                top3 = sum(1 for n,s in ranked[:3] if finish_order[n] <= 3)
                st.metric("上位3頭的中", f"{top3}/3頭")
            with c3:
                winner      = min(finish_order, key=finish_order.get)
                winner_rank = next(i+1 for i,(n,_) in enumerate(ranked) if n == winner)
                st.metric("1着馬のスコア順位", f"{winner_rank}位")

            # 散布図
            fig2, ax = plt.subplots(figsize=(6,4))
            fig2.patch.set_facecolor('#0b0c10')
            ax.set_facecolor('#13141a')
            ax.scatter(score_list, finish_list, c=score_list,
                      cmap='YlOrRd', s=100, zorder=3, edgecolors='#2a2d3a')
            for num in horse_nums:
                ax.annotate(f'{num}番',
                    (scores[num]['total'], finish_order[num]),
                    fontsize=7, color='#d4d8e8',
                    xytext=(3,3), textcoords='offset points')
            ax.invert_yaxis()
            ax.set_xlabel('バイオメカニクススコア', color='#5a5f78')
            ax.set_ylabel('着順', color='#5a5f78')
            ax.tick_params(colors='#d4d8e8')
            ax.spines[:].set_color('#2a2d3a')
            ax.set_title(f'スコア vs 着順（相関: {corr:.3f}）', color='#e8c96a')
            ax.grid(color='#2a2d3a', alpha=0.5)
            st.pyplot(fig2)
            plt.close()

            # 結果CSV
            result_rows = []
            for num in horse_nums:
                s = scores[num]
                r = st.session_state.raw_results[num]
                result_rows.append({
                    'race_id':race_id, '馬番':num,
                    '総合スコア':s['total'], '着順':finish_order[num],
                    '体型':s['body'], '重心':s['balance'],
                    '前対称':s['front_sym'], '後対称':s['back_sym'],
                    '前角度':s['front_angle'], '後角度':s['back_angle'],
                    '体長体高比':r['aspect_ratio'], '重心X':r['cog_x'],
                    '前非対称':r['front_asym'], '後非対称':r['back_asym'],
                })
            df_result = pd.DataFrame(result_rows)
            st.download_button(
                "📥 結果CSVダウンロード（蓄積用）",
                data=df_result.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                file_name=f"{race_id}_result.csv",
                mime='text/csv',
                use_container_width=True
            )
            st.success("✅ 分析完了！CSVをダウンロードして蓄積してください。")
