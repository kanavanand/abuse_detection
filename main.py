import joblib
import streamlit as st
from logging import disable
from scipy.sparse import coo_matrix, hstack


clf = joblib.load('model_files/model_new.joblib')
vec = joblib.load('model_files/vector_new.joblib')
standard_languages = ['Hindi', 'Telugu', 'Marathi', 'Tamil', 'Malayalam', 'Bengali',
       'Kannada', 'Odia', 'Gujarati', 'Haryanvi', 'Bhojpuri', 'Rajasthani',
       'Assamese']
standard_languages = sorted(standard_languages)
map_languages = dict(zip(standard_languages,list(range(len(standard_languages)))))





def process_language(language):
    code = 5
    try:
        code = map_languages[language]
    except:
        code = 5
    vector= [0]*len(map_languages)
    vector[code]=1
    return vector
    



@st.cache(suppress_st_warning=True)
def generate_probab(text,language):
    '''
	Computes the probability of abuse given text and language. 

	Parameters:
	- text (str) : This can be any text in string form
	- language (str) :  Make sure you are passing language from given list 

	Returns : A probab score which tells scale of abuse

	'''
    text = preprocess(text)
    language_vector = process_language(language)
    csr_matrix_vector = coo_matrix(language_vector)

    text_vector = vec.transform([text])
    merged_vector = hstack([text_vector , csr_matrix_vector])
    pred_ = clf.predict_proba(merged_vector)[0][1]
    return pred_


def preprocess(text):
    '''
    Takes string and apply various pre-processing functions

    Parameters:
    -text (str): This will take text

    Return : preprocessed text with cleaned version
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
            summary = generate_probab(my_text,language)
            st.markdown("Probability:-" + str(summary))
                    

    
if __name__ == "__main__":
    st_ui()

