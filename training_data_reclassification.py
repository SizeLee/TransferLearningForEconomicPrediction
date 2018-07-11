import os
import shutil
import re

def remove_dir(rm_dir):
    dir_under = os.listdir(rm_dir)
    for each in dir_under:
        filepath = os.path.join(rm_dir, each)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except:
                print("remove %s error." % filepath)
        elif os.path.isdir(filepath):
            try:
                remove_dir(filepath)
            except:
                print("remove %s error." % filepath)
    os.rmdir(rm_dir)
    return

if __name__ == '__main__':
    nl_level = [10, 30, 64]
    nl_kind_dic = {}
    with open('Area_4km2_Random_100_NL_Statistic_Table.csv', 'r', encoding='gbk') as f:
        attrs = f.readline().strip().split(',')
        # print(attrs)
        sort_setting = 'MEAN'
        sort_attr_index = attrs.index(sort_setting)
        # print(sort_attr_index)
        for line in f:
            # print(line)
            # break
            data = line.strip().split(',')
            if len(data) < 2:
                print('empty line')
                break
            # print(data[0])
            file_name = data[2] + ',' + data[1]
            # print(file_name)
            if file_name in nl_kind_dic:
                print(file_name, data[0])
            nl_value = float(data[sort_attr_index])
            nl_kind = 0
            while(nl_value > nl_level[nl_kind]):
                nl_kind += 1

            ## because classification kind index strat from 1, so here should add 1
            nl_kind += 1
            nl_kind_dic[file_name] = nl_kind

    reclassification_dir = './data'
    if not os.path.exists(reclassification_dir):
        os.mkdir(reclassification_dir)
    else:
        remove_dir(reclassification_dir)
        os.mkdir(reclassification_dir)

    for i in range(len(nl_level)):
        dir_name = 'NL_%d' % (i+1)
        os.mkdir(reclassification_dir + '/' + dir_name)

    pattern = re.compile(r'^\d+.\d+,\d+.\d+')
    rootdir = './NewData0706'
    seconddir = os.listdir(rootdir)
    for each in seconddir:
        secondpath = os.path.join(rootdir, each)
        if os.path.isdir(secondpath):
            countys = os.listdir(secondpath)
            for eachcounty in countys:
                countypath = os.path.join(secondpath, eachcounty)
                for eachfile in os.listdir(countypath):
                    try:
                        filepath = os.path.join(countypath, eachfile)
                        file_mark = re.search(pattern, eachfile).group()
                        # print(file_mark)
                        nl_kind = nl_kind_dic[file_mark]
                        destinypath = './data/NL_%d' % nl_kind
                        shutil.copy(filepath, destinypath)
                    except:
                        print('error in %s' % filepath)


        
        

