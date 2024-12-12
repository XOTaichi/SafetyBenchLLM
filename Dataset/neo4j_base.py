from neo4j import GraphDatabase

class Neo4jHandler:
    def __init__(self):
        # Neo4j 连接配置
        uri = "neo4j://localhost:7687" 
        username = "neo4j"
        password = "Mimashi123!!"

        # 创建连接
        driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver = driver

    def close(self):
        self.driver.close()

    # 插入类别节点（一级、二级、三级），防止同级类别重名
    def insert_category(self, name, level, parent_name=None):
        with self.driver.session() as session:
            result = session.write_transaction(self._insert_category_tx, name, level, parent_name)
            return result

    @staticmethod
    def _insert_category_tx(tx, name, level, parent_name):
        # 检查是否存在同名的类别
        query = """
        MATCH (c:Category {name: $name, level: $level})
        RETURN c
        """
        exists = tx.run(query, name=name, level=level).single()
        if exists:
            return f"Category '{name}' at level {level} already exists."

        # 插入类别节点
        create_query = """
        CREATE (c:Category {name: $name, level: $level})
        RETURN c
        """
        tx.run(create_query, name=name, level=level)

        # 如果有父类别，建立关系
        if parent_name:
            parent_query = """
            MATCH (p:Category {name: $parent_name})
            MATCH (c:Category {name: $name, level: $level})
            CREATE (p)-[:HAS_SUBCATEGORY]->(c)
            """
            tx.run(parent_query, parent_name=parent_name, name=name, level=level)
        return f"Category '{name}' at level {level} inserted successfully."

    # 插入文件节点
    def insert_file(self, file_path):
        with self.driver.session() as session:
            result = session.write_transaction(self._insert_file_tx, file_path)
            return result

    @staticmethod
    def _insert_file_tx(tx, file_path):
        query = """
        CREATE (f:File {path: $file_path})
        RETURN f
        """
        tx.run(query, file_path=file_path)
        return f"File node for '{file_path}' created successfully."

    # 创建关系
    def create_relationship(self, category_name, file_path, relation_type="explicit", subtype=None):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_relationship_tx, category_name, file_path, relation_type, subtype
            )
            return result

    @staticmethod
    def _create_relationship_tx(tx, category_name, file_path, relation_type, subtype):
        if subtype:
            # 如果 subtype 是列表，转换为 Neo4j 的数组
            if isinstance(subtype, list):
                query = f"""
                MATCH (c:Category {{name: $category_name}})
                MATCH (f:File {{path: $file_path}})
                CREATE (c)-[:{relation_type} {{subtype: $subtype}}]->(f)
                """
                tx.run(query, category_name=category_name, file_path=file_path, subtype=subtype)
            else:
                # 如果 subtype 是单个值，正常存储
                query = f"""
                MATCH (c:Category {{name: $category_name}})
                MATCH (f:File {{path: $file_path}})
                CREATE (c)-[:{relation_type} {{subtype: [$subtype]}}]->(f)
                """
                tx.run(query, category_name=category_name, file_path=file_path, subtype=subtype)
        else:
            # 不带 subtype 的关系
            query = f"""
            MATCH (c:Category {{name: $category_name}})
            MATCH (f:File {{path: $file_path}})
            CREATE (c)-[:{relation_type}]->(f)
            """
            tx.run(query, category_name=category_name, file_path=file_path)
        return f"Relationship '{relation_type}' created between Category '{category_name}' and File '{file_path}'."



# 初始化 Neo4jHandler
handler = Neo4jHandler()

# 示例用法

# # 插入一级类别
# print(handler.insert_category("Documents", 1))

# # 插入二级类别
# print(handler.insert_category("Projects", 2, parent_name="Documents"))

# # 插入三级类别
# print(handler.insert_category("Neo4jProject", 3, parent_name="Projects"))

# # 插入文件节点
# print(handler.insert_file("/path/to/your/file.txt"))

# # 建立关系（无 subtype）
# print(handler.create_relationship("Neo4jProject", "/path/to/your/file.txt", relation_type="jailbreak"))

# # 建立关系（有 subtype）
# print(handler.create_relationship("Projects", "/path/to/your/file.txt", relation_type="explicit", subtype="metadata"))

# # 文件节点与二级类别建立关系
# print(handler.create_relationship("Projects", "/path/to/your/file.txt", relation_type="jailbreak"))

