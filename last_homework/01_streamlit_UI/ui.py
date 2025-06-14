import test
import streamlit as st
import ast
from llm import generate_response
import time


def display_homework_page():
    st.title("Python課題サポート")

    # ユーザーが関数を入力
    code = st.text_area("Pythonの関数を入力してください:", height=200)

    if st.button("コードを提出"):
        try:
            tree = ast.parse(code)  # AST解析
            function_def = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            
            if not function_def:
                st.error("関数が見つかりません。")
            else:
                function_name = function_def[0].name  # 最初の関数名を取得
                exec(code, globals())  # 安全なスコープで実行
                function = globals()[function_name]
                
                count,result = test.test_student_function(function)
                st.write(f'{count} passesd')
                if count == 5:
                    st.success("テストに合格しました")
                else:
                    for inp,expected,answer in result:
                        st.write(f'入力: {inp}')
                        st.write(f'期待する出力: {expected} 実際の出力: {answer}')
        except Exception as e:
            st.error(f"エラー: {e}")

def display_chat_page(pipe):
    """チャットページのUIを表示する"""
    st.subheader("質問を入力してください")
    user_question = st.text_area("質問", key="question_input", height=100, value=st.session_state.get("current_question", ""))
    submit_button = st.button("質問を送信")

    # セッション状態の初期化（安全のため）
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    # 質問が送信された場合
    if submit_button and user_question:
        st.session_state.current_question = user_question
        st.session_state.current_answer = "" # 回答をリセット
        st.session_state.feedback_given = False # フィードバック状態もリセット

        with st.spinner("モデルが回答を生成中..."):
            answer, response_time = generate_response(pipe, user_question)
            st.session_state.current_answer = answer
            st.session_state.response_time = response_time
            # ここでrerunすると回答とフィードバックが一度に表示される
            st.rerun()

    # 回答が表示されるべきか判断 (質問があり、回答が生成済みで、まだフィードバックされていない)
    if st.session_state.current_question and st.session_state.current_answer:
        st.subheader("回答:")
        st.markdown(st.session_state.current_answer) # Markdownで表示
        st.info(f"応答時間: {st.session_state.response_time:.2f}秒")

    # フィードバック送信済みの場合、次の質問を促すか、リセットする
    if st.button("次の質問へ"):
        # 状態をリセット
        st.session_state.current_question = ""
        st.session_state.current_answer = ""
        st.session_state.response_time = 0.0
        st.session_state.feedback_given = False
        st.rerun() # 画面をクリア
