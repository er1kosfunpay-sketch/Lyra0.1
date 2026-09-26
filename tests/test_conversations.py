from lyra.conversations import clean_text,detect_lang,valid_messages,duplicate_key

def test_language_and_unicode_cleanup():
 assert detect_lang('Привет, мир!')=='ru'; assert detect_lang('Hello there')=='en'
 assert clean_text('  привет\x00  мир ')=='привет мир'

def test_short_reply_context_validation():
 assert valid_messages([{'role':'user','content':'Привет!'}, {'role':'assistant','content':'Привет!'}])
 assert valid_messages([{'role':'user','content':'Ты придёшь?'},{'role':'assistant','content':'Да'}])
 assert not valid_messages([{'role':'user','content':'x'},{'role':'assistant','content':'ok'}])

def test_near_duplicate_fingerprint_normalizes_format():
 a=[{'role':'user','content':'Hello, there!'}, {'role':'assistant','content':'Nice to meet you.'}]
 b=[{'role':'user','content':'Hello there'}, {'role':'assistant','content':'Nice to meet you'}]
 assert duplicate_key(a)==duplicate_key(b)
