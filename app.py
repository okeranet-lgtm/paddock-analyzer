import streamlit as st
import cv2
import numpy as np
import pandas as pd
import zipfile
import os
import io
import re
import shutil
import time
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

st.set_page_config(
    page_title="🏇 Paddock Analyzer",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ━━━ スタイル ━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+JP:wght@300;400;700&display=swap');
:root {
    --gold: #c9a84c; --gold2: #e8c96a;
    --dark: #0b0c10; --dark2: #13141a; --dark3: #1c1e28;
    --green: #1db954; --red: #e53935; --blue: #2979ff;
    --text: #d4d8e8; --muted: #5a5f78;
}
.stApp { background: #0b0c10; color: #d4d8e8; }
h1 { font-family: 'Bebas Neue', sans-serif !important; 
     background: linear-gradient(135deg, #c9a84c, #e8c96a);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     letter-spacing: .12em; font-size: 2.4rem !important; }
h2, h3 { font-family: 'Noto Sans JP', sans-serif !important; color: #e8c96a !important; }
.stButton > button {
    background: linear-gradient(135deg, #1a6b35, #1db954) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
}
.metric-card {
    background: #13141a; border: 1px solid #2a2d3a;
    border-radius: 10px; padding: 12px 16px; margin: 4px 0;
}
.rank-gold   { color: #FFD700; font-weight: 700; font-size: 1.1rem; }
.rank-silver { color: #C0C0C0; font-weight: 700; font-size: 1.1rem; }
.rank-bronze { color: #CD7F32; font-weight: 700; font-size: 1.1rem; }
.rank-other  { color: #d4d8e8; font-size: 1rem; }
.stProgress > div > div { background: linear-gradient(90deg, #c9a84c, #1db954) !important; }
div[data-testid="stFileUploader"] {
    background: #13141a; border: 2px dashed #2a2d3a;
    border-radius: 12px; padding: 8px;
}
.stTabs [data-baseweb="tab-list"] { background: #13141a; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #5a5f78 !important; font-weight: 700; }
.stTabs [aria-selected="true"] { color: #e8c96a !important; }
</style>
""", unsafe_allow_html=True)

# ━━━ セッション初期化 ━━━
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'scores' not in st.session_state:
    st.session_state.scores = {}
if 'raw_results' not in st.session_state:
    st.session_state.raw_results = {}
if 'race_id' not in st.session_state:
    st.session_state.race_id = ''

# ━━━ ヘッダー ━━━
st.markdown("# 🏇 PADDOCK ANALYZER")
st.markdown("##### 競走馬バイオメカニクス解析・ランキングシステム")
st.divider()

# ━━━ タブ ━━━
tab1, tab2, tab3 = st.tabs(["📁 ZIPアップロード・解析", "📊 ランキング結果", "📈 データ蓄積・精度改善"])

# ━━━━━━━━━━━━━━━━━━━━
# TAB 1: アップロード・解析
# ━━━━━━━━━━━━━━━━━━━━
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
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

    with col_right:
        st.markdown("### 📁 ZIPファイルアップロード")
        st.caption("ファイル名形式: `horse_01_1.jpg` / `horse_01_2.jpg` など")
        uploaded_zip = st.file_uploader(
            "ZIPファイルをドロップまたは選択",
            type=['zip'],
            label_visibility="collapsed"
        )

        if uploaded_zip:
            st.success(f"✅ {uploaded_zip.name} ({uploaded_zip.size//1024}KB)")

    st.divider()

    # 解析実行
    if uploaded_zip:
        if st.button("🔬 骨格解析・ランキング生成", use_container_width=True):
            with st.spinner("解析中... しばらくお待ちください"):
                try:
                    # ━━━ ZIP展開 ━━━
                    tmp_dir    = tempfile.mkdtemp()
                    upload_dir = os.path.join(tmp_dir, "uploads")
                    work_dir   = os.path.join(tmp_dir, "work")
                    os.makedirs(upload_dir); os.makedirs(work_dir)

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

                    # ━━━ 動画作成 ━━━
                    sorted_nums = sorted(horse_map.keys())
                    frame_list  = [(n, p) for n in sorted_nums for p in horse_map[n]]
                    first_img   = cv2.imread(frame_list[0][1])
                    h, w        = first_img.shape[:2]
                    video_path  = os.path.join(work_dir, "horses.mp4")
                    writer      = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w,h))
                    for _, fpath in frame_list:
                        frame = cv2.imread(fpath)
                        if frame is not None:
                            writer.write(cv2.resize(frame, (w,h)))
                    writer.release()

                    # ━━━ DeepLabCut 骨格検出 ━━━
                    progress = st.progress(0, text="モデルロード中...")
                    from deeplabcut.modelzoo.video_inference import video_inference_superanimal, download_huggingface_model

                    progress.progress(20, text="SuperAnimalモデルダウンロード中...")
                    download_huggingface_model("superanimal_quadruped")

                    progress.progress(40, text="骨格検出中（CPU処理・数分かかります）...")
                    video_inference_superanimal(
                        videos=[video_path],
                        superanimal_name="superanimal_quadruped",
                        model_name="resnet_50",
                        detector_name="fasterrcnn_resnet50_fpn_v2",
                        dest_folder=work_dir,
                    )

                    progress.progress(80, text="バイオメカニクス計算中...")

                    # ━━━ H5読み込み ━━━
                    h5_file = [f for f in os.listdir(work_dir) if f.endswith('.h5')][0]
                    df = pd.read_hdf(os.path.join(work_dir, h5_file))
                    scorer      = df.columns.get_level_values(0)[0]
                    individuals = df[scorer].columns.get_level_values(0).unique()

                    def get_kp(fi, ind, part):
                        try:
                            x  = df[scorer][ind][part]['x'].iloc[fi]
                            y  = df[scorer][ind][part]['y'].iloc[fi]
                            lk = df[scorer][ind][part]['likelihood'].iloc[fi]
                            return (float(x), float(y), float(lk))
                        except: return (None, None, 0.0)

                    def best_ind(fi):
                        key_parts = ['back_base','back_end','front_left_paw','back_right_paw','neck_base']
                        best, best_s = None, 0
                        for ind in individuals:
                            s = sum(get_kp(fi, ind, p)[2] for p in key_parts)
                            if s > best_s: best_s=s; best=ind
                        return best if best_s > 1.0 else None

                    def calc_angle(p1, p2, p3):
                        if None in list(p1)+list(p2)+list(p3): return None
                        v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
                        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
                        cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
                        return round(float(np.degrees(np.arccos(np.clip(cos,-1,1)))), 1)

                    def analyze_frame(fi):
                        ind = best_ind(fi)
                        if not ind: return None
                        g = lambda p: get_kp(fi, ind, p)
                        neck=g('neck_base'); tail=g('tail_base')
                        back_b=g('back_base'); belly=g('belly_bottom'); back_m=g('back_middle')
                        fl_t=g('front_left_thai');  fl_k=g('front_left_knee');  fl_p=g('front_left_paw')
                        fr_t=g('front_right_thai'); fr_k=g('front_right_knee'); fr_p=g('front_right_paw')
                        bl_t=g('back_left_thai');   bl_k=g('back_left_knee');   bl_p=g('back_left_paw')
                        br_t=g('back_right_thai');  br_k=g('back_right_knee');  br_p=g('back_right_paw')
                        blen = np.hypot(neck[0]-tail[0],neck[1]-tail[1]) if None not in [neck[0],tail[0]] else None
                        bhei = np.hypot(back_b[0]-belly[0],back_b[1]-belly[1]) if None not in [back_b[0],belly[0]] else None
                        fl_a=calc_angle(fl_t[:2],fl_k[:2],fl_p[:2]); fr_a=calc_angle(fr_t[:2],fr_k[:2],fr_p[:2])
                        bl_a=calc_angle(bl_t[:2],bl_k[:2],bl_p[:2]); br_a=calc_angle(br_t[:2],br_k[:2],br_p[:2])
                        return {
                            'aspect_ratio': round(blen/bhei,2) if blen and bhei and bhei>0 else None,
                            'cog_x': round((back_m[0]-neck[0])/(tail[0]-neck[0]+1e-8),2) if None not in [back_m[0],neck[0],tail[0]] else None,
                            'fl_angle':fl_a,'fr_angle':fr_a,'bl_angle':bl_a,'br_angle':br_a,
                            'front_asym': round(abs(fl_a-fr_a),1) if fl_a and fr_a else None,
                            'back_asym':  round(abs(bl_a-br_a),1) if bl_a and br_a else None,
                        }

                    # 馬番ごと平均化
                    raw_results = {}
                    fi = 0
                    for num in sorted_nums:
                        frame_data = []
                        for _ in horse_map[num]:
                            r = analyze_frame(fi)
                            if r: frame_data.append(r)
                            fi += 1
                        if not frame_data: continue
                        avg = {}
                        for key in frame_data[0].keys():
                            vals = [d[key] for d in frame_data if d[key] is not None]
                            avg[key] = round(float(np.mean(vals)),2) if vals else None
                        raw_results[num] = avg

                    # スコア計算
                    def calc_score(r):
                        s = {}
                        ar=r['aspect_ratio']
                        s['body']       = 10 if ar and 1.6<=ar<=2.0 else 7 if ar and 1.4<=ar<=2.2 else 5 if ar else 5
                        cx=r['cog_x']
                        s['balance']    = 10 if cx and 0.45<=cx<=0.55 else 7 if cx and 0.40<=cx<=0.60 else 4 if cx else 5
                        fa=r['front_asym']
                        s['front_sym']  = 10 if fa and fa<5 else 8 if fa and fa<15 else 5 if fa and fa<30 else 3 if fa and fa<50 else 1 if fa else 5
                        ba=r['back_asym']
                        s['back_sym']   = 10 if ba and ba<5 else 8 if ba and ba<15 else 5 if ba and ba<25 else 3 if ba and ba<40 else 1 if ba else 5
                        fl=r['fl_angle']; fr=r['fr_angle']
                        af=(fl+fr)/2 if fl and fr else None
                        s['front_angle']= 10 if af and 155<=af<=175 else 7 if af and 145<=af<=180 else 4 if af else 5
                        bl=r['bl_angle']; br=r['br_angle']
                        ab=(bl+br)/2 if bl and br else None
                        s['back_angle'] = 10 if ab and 150<=ab<=170 else 7 if ab and 140<=ab<=175 else 4 if ab else 5
                        w={'body':1.5,'balance':2.0,'front_sym':2.5,'back_sym':2.5,'front_angle':1.5,'back_angle':1.5}
                        s['total']=round(sum(s[k]*w[k] for k in w)/sum(10*w[k] for k in w)*100,1)
                        return s

                    scores = {num: calc_score(r) for num,r in raw_results.items()}
                    st.session_state.scores      = scores
                    st.session_state.raw_results = raw_results
                    st.session_state.analyzed    = True
                    st.session_state.race_id     = race_id

                    progress.progress(100, text="✅ 解析完了！")
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    st.success(f"✅ {len(scores)}頭の解析が完了しました！「📊 ランキング結果」タブを確認してください。")

                except Exception as e:
                    st.error(f"❌ エラーが発生しました: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 ZIPファイルをアップロードして解析を開始してください")

# ━━━━━━━━━━━━━━━━━━━━
# TAB 2: ランキング結果
# ━━━━━━━━━━━━━━━━━━━━
with tab2:
    if not st.session_state.analyzed:
        st.info("📁 まずZIPファイルをアップロードして解析を実行してください")
    else:
        scores      = st.session_state.scores
        raw_results = st.session_state.raw_results
        race_id     = st.session_state.race_id
        ranked      = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        medals      = {1:"🥇",2:"🥈",3:"🥉"}

        st.markdown(f"### 🏇 {race_id} バイオメカニクスランキング")

        # ━━━ ランキング表示 ━━━
        col_rank, col_chart = st.columns([1, 1])

        with col_rank:
            st.markdown("#### 📋 総合ランキング")
            for rank, (num, s) in enumerate(ranked, 1):
                medal = medals.get(rank, f"{rank}位")
                color = ["rank-gold","rank-silver","rank-bronze"] if rank <= 3 else ["rank-other"]
                css   = color[0] if rank <= 3 else color[0]
                bar_w = int(s['total'])
                st.markdown(f"""
                <div class="metric-card">
                  <span class="{css}">{medal} {rank}位 &nbsp; {num}番馬</span>
                  <span style="float:right;color:#e8c96a;font-weight:700;font-size:1.1rem">{s['total']}点</span>
                  <div style="background:#2a2d3a;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:linear-gradient(90deg,#c9a84c,#1db954);width:{bar_w}%;height:6px;border-radius:4px"></div>
                  </div>
                  <div style="font-size:.7rem;color:#5a5f78;margin-top:4px">
                    体型:{s['body']} 重心:{s['balance']} 前対称:{s['front_sym']} 後対称:{s['back_sym']} 前角:{s['front_angle']} 後角:{s['back_angle']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_chart:
            st.markdown("#### 📊 スコアチャート")
            fig, axes = plt.subplots(1, 2, figsize=(10, max(5, len(ranked)*0.45+1)))
            fig.patch.set_facecolor('#0b0c10')

            # 棒グラフ
            ax1 = axes[0]
            ax1.set_facecolor('#13141a')
            nums_r   = [n for n,_ in ranked]
            totals_r = [s['total'] for _,s in ranked]
            bar_cols = ['#FFD700','#C0C0C0','#CD7F32'] + ['#2979ff']*(len(ranked)-3)
            bars = ax1.barh([f"#{n}" for n in nums_r], totals_r, color=bar_cols, height=0.65)
            ax1.set_xlim(0,105)
            ax1.set_facecolor('#13141a')
            ax1.tick_params(colors='#d4d8e8', labelsize=9)
            ax1.spines[:].set_color('#2a2d3a')
            ax1.set_xlabel('Score', color='#5a5f78', fontsize=8)
            for i,(n,v) in enumerate(zip(nums_r,totals_r)):
                ax1.text(v+0.5,i,f'{v}',va='center',color='#d4d8e8',fontsize=8)
            ax1.set_title('Ranking', color='#e8c96a', fontsize=10, pad=8)

            # レーダー（上位4頭）
            cats = ['Body','Balance','FrontSym','BackSym','FrontAng','BackAng']
            keys = ['body','balance','front_sym','back_sym','front_angle','back_angle']
            angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()
            angles += angles[:1]
            ax2 = fig.add_subplot(122, polar=True)
            ax2.set_facecolor('#13141a')
            radar_cols = ['#FFD700','#C0C0C0','#CD7F32','#2979ff']
            for i,(num,s) in enumerate(ranked[:4]):
                vals = [s[k] for k in keys]+[s[keys[0]]]
                ax2.plot(angles,vals,'o-',color=radar_cols[i],label=f'#{num}',linewidth=2)
                ax2.fill(angles,vals,alpha=0.08,color=radar_cols[i])
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(cats,fontsize=7,color='#d4d8e8')
            ax2.set_ylim(0,10)
            ax2.tick_params(colors='#5a5f78',labelsize=7)
            ax2.spines['polar'].set_color('#2a2d3a')
            ax2.set_facecolor('#13141a')
            ax2.grid(color='#2a2d3a')
            ax2.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),fontsize=8,
                       facecolor='#13141a',edgecolor='#2a2d3a',labelcolor='#d4d8e8')
            ax2.set_title('Top4 Radar', color='#e8c96a', fontsize=10, pad=15)
            axes[1].remove()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()

        # ━━━ 詳細テーブル ━━━
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
        st.dataframe(
            df_disp, use_container_width=True, hide_index=True,
            column_config={
                '総合': st.column_config.ProgressColumn('総合', min_value=0, max_value=100),
            }
        )

        # ━━━ CSV ダウンロード ━━━
        csv = df_disp.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv.encode('utf-8-sig'),
            file_name=f"{race_id}_ranking.csv",
            mime='text/csv',
            use_container_width=True
        )

# ━━━━━━━━━━━━━━━━━━━━
# TAB 3: データ蓄積・精度改善
# ━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### 📈 レース結果との照合・精度改善")
    st.info("解析後に実際の着順を入力すると、スコアとの相関を分析して精度改善に活用できます。")

    if not st.session_state.analyzed:
        st.warning("まず解析を実行してください")
    else:
        scores  = st.session_state.scores
        race_id = st.session_state.race_id
        ranked  = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        horse_nums = sorted(scores.keys())

        st.markdown(f"#### 🏁 {race_id} 着順入力")
        st.caption("実際のレース着順を入力してください")

        cols = st.columns(4)
        finish_order = {}
        for i, num in enumerate(horse_nums):
            with cols[i % 4]:
                order = st.number_input(
                    f"{num}番", min_value=1, max_value=len(horse_nums),
                    value=i+1, key=f"order_{num}"
                )
                finish_order[num] = order

        if st.button("📊 相関分析・保存", use_container_width=True):
            # 相関分析
            score_list  = [scores[n]['total'] for n in horse_nums]
            finish_list = [finish_order[n] for n in horse_nums]
            corr = np.corrcoef(score_list, finish_list)[0,1]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("スコアと着順の相関", f"{corr:.3f}",
                          help="-1に近いほど「高スコア=上位着順」の精度が高い")
            with col2:
                top3_correct = sum(1 for n,s in ranked[:3] if finish_order[n] <= 3)
                st.metric("上位3頭的中数", f"{top3_correct}/3頭")
            with col3:
                winner = min(finish_order, key=finish_order.get)
                winner_rank = next(i+1 for i,(n,_) in enumerate(ranked) if n==winner)
                st.metric("1着馬のスコア順位", f"{winner_rank}位")

            # 散布図
            fig2, ax = plt.subplots(figsize=(6,4))
            fig2.patch.set_facecolor('#0b0c10')
            ax.set_facecolor('#13141a')
            sc = ax.scatter(score_list, finish_list, c=score_list,
                           cmap='YlOrRd', s=100, zorder=3, edgecolors='#2a2d3a')
            for num in horse_nums:
                ax.annotate(f'{num}番',
                    (scores[num]['total'], finish_order[num]),
                    fontsize=7, color='#d4d8e8',
                    xytext=(3,3), textcoords='offset points')
            ax.set_xlabel('バイオメカニクススコア', color='#5a5f78')
            ax.set_ylabel('着順', color='#5a5f78')
            ax.invert_yaxis()
            ax.tick_params(colors='#d4d8e8')
            ax.spines[:].set_color('#2a2d3a')
            ax.set_title(f'スコア vs 着順 (相関: {corr:.3f})', color='#e8c96a')
            ax.grid(color='#2a2d3a', alpha=0.5)
            st.pyplot(fig2)
            plt.close()

            # 結果CSV保存
            result_rows = []
            for num in horse_nums:
                s = scores[num]
                r = st.session_state.raw_results[num]
                result_rows.append({
                    'race_id':race_id, '馬番':num,
                    '総合スコア':s['total'], '着順':finish_order[num],
                    '体型':s['body'],'重心':s['balance'],
                    '前対称':s['front_sym'],'後対称':s['back_sym'],
                    '前角度':s['front_angle'],'後角度':s['back_angle'],
                    '体長体高比':r['aspect_ratio'],'重心X':r['cog_x'],
                    '前非対称':r['front_asym'],'後非対称':r['back_asym'],
                })
            df_result = pd.DataFrame(result_rows)
            csv_result = df_result.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 結果CSVダウンロード（蓄積用）",
                data=csv_result.encode('utf-8-sig'),
                file_name=f"{race_id}_result.csv",
                mime='text/csv',
                use_container_width=True
            )
            st.success("✅ 分析完了！CSVをダウンロードして蓄積してください。")
