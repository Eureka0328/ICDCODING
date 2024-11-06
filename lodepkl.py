import pickle

# 加载变量
with open('result_y.pkl', 'rb') as file:
    variables = pickle.load(file)

yhat = variables['yhat']
y = variables['y']
yhat_raw = variables['yhat_raw']

print(f"yhat_raw:{yhat_raw[0:4]}")


with open('code.pkl', 'rb') as file1:
    variables = pickle.load(file1)

code_or = variables['ind2c']
code = []

for value in code_or.values():
    code.append(value)

print(code_or)