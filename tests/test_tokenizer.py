from lyra.tokenizer import LyraTokenizer

def test_tokenizer_roundtrip(tmp_path):
 corpus=tmp_path/'data.txt'; corpus.write_text('Привет, мир! Hello world. '+('game:GetService("Players")\n'*10),encoding='utf8')
 tok=LyraTokenizer.train([corpus],vocab_size=512)
 for s in ['Привет, как дела?','Hello, world!','你好 🌍','game:GetService("Players")']:
  assert tok.decode(tok.encode(s))==s

def test_special_tokens_are_single_ids(tmp_path):
 corpus=tmp_path/'data.txt'; corpus.write_text(('assistant <ASSISTANT> user\n'*10),encoding='utf8'); tok=LyraTokenizer.train([corpus],512)
 assert tok.encode('<ASSISTANT>')==[tok.id('<ASSISTANT>')]
