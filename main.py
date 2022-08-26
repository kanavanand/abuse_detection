from logging import disable
import joblib
import streamlit as st

clf = joblib.load('model.joblib')
vec = joblib.load('vector.joblib')


@st.cache(suppress_st_warning=True)
def generate_summary(text,language):
    '''
	Computes the probability of abuse given text and language. 

	Parameters:
	- text (str) : This can be any text in string form
	- language (str) :  Make sure you are passing language from given list 

	Returns : A probab score which tells scale of abuse

	'''
    text = preprocess(text)
    pred_ = clf.predict_proba(vec.transform([text]))[0][1]
    return pred_


def preprocess(text):
    '''
    Takes string and apply various pre-processing functions
    '''
    return text



################################## UI ##############################################
def st_ui():
    '''
    Streamlit UI
    '''
    st.set_page_config(
        page_title="Abuse detection-indian-languages",
        page_icon="🍲",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.write("# Multi-Lingual Abuse Detection")
    with st.expander("ℹ️ - About this app", expanded=True):
        st.write(
            """     
    -   The *Abuse detectionr* app is an easy-to-use interface built in Streamlit for detecting abuse in multiple-indian-languages.
    -   It is using sklearn models to make predictions and can widely be used in detecting abuse on twitter or any other type of social media.
            """
        )

    st.markdown("")

    st.markdown(
        """
            This Daisi allows you to detect the abuse detection in indian Languages
        """
    )
    col1, col2 = st.columns([6, 4])
    with col1:
        st.markdown("Select the Language", unsafe_allow_html=True)
        language = st.selectbox("Select the Language", index=0, options=["English",'Hindi', 'Telugu', 'Marathi', 'Tamil', 'Malayalam', 'Bengali',
       'Kannada', 'Odia', 'Gujarati', 'Haryanvi', 'Bhojpuri', 'Rajasthani',
       'Assamese'])
        st.markdown("Type the comment in below box")
        my_text = st.text_area("", "what the fuck is wrong with you?", key='text_key')
    with col2:
        st.markdown("### 🔑 Languages supported")
        with st.expander("Where did this story start?", expanded=True):
            # st.markdown(meta.STORY, unsafe_allow_html=True)
            st.write("""
        - English
        - Hindi
        - Punjabi
        - Telugu
        - Marathi 
        - Tamil 
        - Malayalam 
        - Bengali
        - Kannada 
        - Odia
        - Gujarati
        - Haryanvi
        - Bhojpuri
        - Rajasthani
        - Assamese'
        """,disabled=True)
        # sum_text.text_area("", "",  disabled=True)
        

    generate_btn = st.button('Generate')

    st.markdown(
            "<hr />",
            unsafe_allow_html=True
        )

    if generate_btn:        
        with st.spinner("Generating recipe..."):
            st.title("Results - ")
            summary = generate_summary(my_text,my_text)
            st.markdown("Probability:-" + str(summary))
                    

    
if __name__ == "__main__":
    st_ui()

