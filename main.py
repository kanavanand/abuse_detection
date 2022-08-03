import joblib
import streamlit as st

clf = joblib.load('model.joblib')
vec = joblib.load('vector.joblib')


@st.cache(suppress_st_warning=True)
def generate_summary(text):
    """
    Generate probablity given text
    """
    pred_ = clf.predict_proba(vec.transform([text]))[0][1]
    return pred_

################################## UI ##############################################

def st_ui():
    st.write("# Welcome to the abuse detection Daisi!")
    st.markdown(
        """
            This daisi allows you to obtain a abuse detection of text.
        """
    )
    col1, col2 = st.columns([1,1])
    with col1:
        st.title("Text")
        my_text = st.text_area("", "what the fuck is wrong with you?", height=300, key='text_key')
    with col2:
        st.title("Abuse detection")
        sum_text = st.empty()
        sum_text.text_area("", "", height=300, disabled=True)
        
    generate_btn = st.button('Generate')
    if generate_btn:
        sum_text.text_area("", "Generating...", height=300, disabled=True)
        summary = generate_summary(my_text)
        sum_text.text_area("", summary, height=300)
                

    
if __name__ == "__main__":
    st_ui()

