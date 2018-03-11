

# if __name__ == '__main__':
#     with open('train_contents.txt','r',encoding="utf8")as tc:
#         with open('train_labels.txt','r',encoding="utf8")as tl:
#             for c,l in zip(tc,tl):
#                 line = ''.join([c.strip(),' __label__',l])
#                 with open('train.txt', 'a',encoding="utf8")as all:
#                     all.write(line)
#     print('done')

# if __name__ == '__main__':
#     with open('test_contents.txt','r',encoding="utf8")as tc:
#         with open('test_labels.txt','r',encoding="utf8")as tl:
#             for c,l in zip(tc,tl):
#                 line = ''.join([c.strip(),' __label__',l])
#                 with open('test.txt', 'a',encoding="utf8")as all:
#                     all.write(line)
#     print('done')

if __name__ == '__main__':
    from pyfasttext import fasttext

    model = FastText()
    clf = model.supervised('train.txt','fasttext.model',label_prefix="__label__",epoch=2)
    result = model.test("test.txt")
    pred = model.predict_file('test.txt', k=2)
    print (result.precision)
    print (result.recall)