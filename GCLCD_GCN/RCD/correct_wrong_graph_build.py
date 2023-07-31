import json

def build_local_map():
    data_file = '../data/junyi/train_set.json'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    u_from_e_correct = '' # e(src) to k(dst) when score is 1
    e_from_u_correct = '' # k(src) to k(dst) when score is 1

    u_from_e_incorrect = '' # e(src) to k(dst) when score is 0
    e_from_u_incorrect = '' # k(src) to k(dst) when score is 0

    print (len(data))
    for line in data:
        exer_id = line['exer_id'] - 1
        user_id = line['user_id'] - 1
        score = line['score']
        for k in line['knowledge_code']:
            if score == 1:
                u_from_e_correct += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
                e_from_u_correct += str(user_id + exer_n) + '\t' + str(exer_id) + '\n'
            else:
                u_from_e_incorrect += str(exer_id) + '\t' + str(user_id + exer_n) + '\n'
                e_from_u_incorrect += str(user_id + exer_n) + '\t' + str(exer_id) + '\n'

    with open('../data/junyi/graph/u_from_e_correct.txt', 'w') as f:
        f.write(u_from_e_correct)
    with open('../data/junyi/graph/e_from_u_correct.txt', 'w') as f:
        f.write(e_from_u_correct)

    with open('../data/junyi/graph/u_from_e_incorrect.txt', 'w') as f:
        f.write(u_from_e_incorrect)
    with open('../data/junyi/graph/e_from_u_incorrect.txt', 'w') as f:
        f.write(e_from_u_incorrect)

if __name__ == '__main__':
    build_local_map()
