import streamlit as st
import test
import ui
from config import MODEL_NAME
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import HfFolder
import llm

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")


@st.cache_resource
def load_model():
    """量子化された LLM モデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")  # 使用デバイスを表示

        # 4-bit 量子化の設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # GPU メモリが厳しければ `float32` に変更
        )

        # トークナイザのロード
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 量子化されたモデルの読み込み
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # ✅ 量子化設定を適用
            device_map="auto"  # ✅ `auto` で適切なデバイスに割り当て
        )

        # `pipeline()` には `device` を渡さず、量子化を考慮
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}  # ✅ 量子化対応
        )

        st.success(f"量子化モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

# モデルをロード
pipe = load_model()

## ページ設定
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "課題" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["課題", "LLM_chat"],
    key="page_selector",
    index=["課題", "LLM_chat"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)

# 課題提出page
if st.session_state.page == "課題":
    ui.display_homework_page()


# LLM_chatページ
if st.session_state.page == "LLM_chat":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")