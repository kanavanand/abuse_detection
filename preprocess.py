import re



standard_languages = ['Hindi', 'Telugu', 'Marathi', 'Tamil', 'Malayalam', 'Bengali',
       'Kannada', 'Odia', 'Gujarati', 'Haryanvi', 'Bhojpuri', 'Rajasthani',
       'Assamese']
standard_languages = sorted(standard_languages)
map_languages = dict(zip(standard_languages,list(range(len(standard_languages)))))



def replace_phone_number(text):
    '''
    Regex pattern to replace phone number with MOBILE_NUMBER
    Objective is to standardize all the phone number patterns
    '''
    find_all_mobile_number = re.compile(r'(?:(?:\+|0{0,2})91(\s*[\ -]\s*)?|[0]?)?[789]\d{9}|(\d[ -]?){10}\d$')
    return re.sub(pattern= find_all_mobile_number, repl='MOBILE_NUMBER', string=text)


def replace_user_mention(text):
    '''
    Regex pattern to replace usermention with USER_MENTION
    Objective is to standardize all the usermention patterns
    '''
    find_all_user_mention = re.compile(r'@\w*')
    return re.sub(pattern= find_all_user_mention, repl='USER_MENTION', string=text)


def replace_url(text):
    '''
    Regex pattern to replace urls with URL
    Objective is to standardize all the URL patterns
    '''
    find_all_url = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')
    return re.sub(pattern= find_all_url, repl='URL', string=text)


def replace_class_mention(text):
    '''
    Regex pattern to replace class mention with CLASS_TH
    Objective is to standardize all the class mention  patterns
    Example: 10th , 1st -> to class mention
    '''

    find_all_class = re.compile(r'[0-9]+(?:st|[nr]d|th)')
    return re.sub(pattern= find_all_class, repl='CLASS_TH', string=text)


def replace_salary_mention(text):
    '''
    Regex pattern to replace class mention with CLASS_TH
    Objective is to standardize all the class mention  patterns
    Example: 10k,50k -> to salary mention
    '''

    find_salary_mention = re.compile(r'[0-9]+(?:k)')
    return re.sub(pattern= find_salary_mention, repl='SALARY_MENTION', string=text)


def preprocess_text(text):
    '''
    Takes string and apply various pre-processing functions

    Parameters:
    -text (str): This will take text

    Return : preprocessed text with cleaned version
    '''
    text = replace_phone_number(text)
    text = replace_class_mention(text)
    text = replace_url(text)
    text = replace_class_mention(text) 
    text = replace_salary_mention(text)
    return text



def process_language(language):
    code = 5
    try:
        code = map_languages[language]
    except:
        code = 5
    vector= [0]*len(map_languages)
    vector[code]=1
    return vector
    
