import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://github.com/alchemistjp/ai_app_t1/dx01.jpg");
    }
   </style>
    """,
    unsafe_allow_html=True
)





st.sidebar.title("報告版_AI搭載アプリ(画像分類）")
st.sidebar.write("CIFAR-10データを学習し10分類に分けたCNNモデルを使って、画像の要素割合をグラフ表示するアプリ")

st.sidebar.write("")

img_source = st.sidebar.radio("要素解析する画像を選択してください。",
                              ("コンピュータからアップロード", "PCのカメラで撮影"))
if img_source == "コンピュータからアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "PCのカメラで撮影":
    img_file = st.camera_input("PCのカメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        # 予測
        results = predict(img)

        # result
        st.subheader("解析結果")
        n_top = 3  # 割合が高い順に3位まで返す
        for result in results[:n_top]:
            st.write(str(round(result[2]*100, 2)) + "%の割合で" + result[0] + "の要素が見つかりました。")

        # 円グラフの表示
        pie_labels = [result[1] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[2] for result in results[:n_top]]
        pie_probs.append(sum([result[2] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  # 円グラフ
        st.pyplot(fig)
