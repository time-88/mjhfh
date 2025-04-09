from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# 加载预训练的GPT-2模型和tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@app.route('/chat', methods=['POST'])
def chat():
    # 获取前端发送的消息
    user_input = request.json.get('message')

    # 使用GPT-2生成回复
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回AI的回复
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
