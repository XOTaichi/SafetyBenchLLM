import argparse
from neo4j_base import Neo4jHandler
'''
python add.py one "犯罪(Crimes)"
python add.py two "恐怖主义(Terror)"
python add.py three "造炸弹(Bomb)"
python add.py --type file --name "文件名"
python add.py --type relation --name "类别名" --name2 "文件名" --subtype "explicit"
python add.py --type relation --name "类别名" --name2 "文件名" --subtype "jailbreak"
'''
def main():
    parser = argparse.ArgumentParser(description='Add a new node or relationship to the graph database.')
    
    # 共享参数，所有节点都需要
    parser.add_argument('type', type=str, help='The type of the new node or relationship.')
    parser.add_argument('name', type=str, help='The name of the new node.')
    parser.add_argument('name2', type=str, help='The name of the second node (for relationships)', nargs='?', default=None)
    parser.add_argument('subtype', type=str, help='The subtype of the new node or relationship.', nargs='?', default=None)
    
    args = parser.parse_args()

    handler = Neo4jHandler()
    
    node_type = args.type
    node_name = args.name
    node_name2 = args.name2
    node_subtype = args.subtype

    # 插入一级类别
    if node_type == 'one':
        handler.insert_category(node_name, 1) 
    # 插入二级类别
    elif node_type == 'two':
        if node_subtype is None:
            raise ValueError("二级节点需要一个父节点属性（subtype）")
        handler.insert_category(node_name, 2, node_subtype)
    # 插入三级类别
    elif node_type == 'three':
        if node_subtype is None:
            raise ValueError("三级节点需要一个父节点属性（subtype）")
        handler.insert_category(node_name, 3, node_subtype)
    # 插入文件位置
    elif node_type == 'file':
        handler.insert_file(node_name)
    # 插入文件和类别之间的关系
    elif node_type == 'relation':
        if node_name2 is None:
            raise ValueError("关系节点需要两个节点属性（name 和 name2）")
        if node_subtype == "explicit":
            handler.create_relationship(node_name, node_name2, node_subtype)
        elif node_subtype == "jailbreak":
            subtype = input("请输入越狱类型，以英文,分割：")
            subtype = subtype.split(',')
            handler.create_relationship(node_name, node_name2, node_subtype, subtype)
        else:
            raise ValueError("未知的关系类型")
    else:
        raise ValueError("未知的节点类型")

if __name__ == "__main__":
    main()
