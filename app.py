from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 初始化Flask应用
app = Flask(__name__)

# 加载预训练的GPT-2模型和tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 确保模型在GPU上运行，如果GPU不可用，则在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@app.route('/chat', methods=['POST'])
def chat():
    # 从请求中获取用户消息
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # 对用户输入进行tokenize
    inputs = tokenizer.encode(user_input, return_tensors='pt').to(device)

    # 使用模型生成回复
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

    # 解码生成的回复
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回AI的回复
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
