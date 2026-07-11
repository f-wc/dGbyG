import time
import sqlite3
import pandas as pd
from pathlib import Path

from ..config import config


def pubchem_gz_to_db(source_file, target_file, chunksize: int = 1000000):
    
    tabel_name = 'pubchem'
    source_file = Path(config.pubchem_database_path) / source_file
    target_file = Path(config.pubchem_database_path) / target_file

    # 分块读取压缩文件
    chunk_iter = pd.read_csv(
        source_file,
        sep='\t',
        compression='gzip',
        header=None,
        names=['CID', 'SMILES'],
        dtype={'CID': int, 'SMILES': str},
        chunksize=chunksize
    )

    # 连接 SQLite（启用 WAL 模式提升写入性能）
    conn = sqlite3.connect(target_file)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能与安全

    # 创建表：CID 设为主键（自动创建索引，且强制唯一）
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {tabel_name} (
            CID INTEGER PRIMARY KEY,
            SMILES TEXT NOT NULL
        )
    """)
    
    # 准备插入语句（使用 ? 占位符）
    insert_sql = f"INSERT OR REPLACE INTO {tabel_name} (CID, SMILES) VALUES (?, ?)"

    start_time = time.time()
    total_rows = 0
    for i, chunk in enumerate(chunk_iter, 1):
        # 将 DataFrame 转换为元组列表，避免逐行迭代
        data = list(chunk.itertuples(index=False, name=None))
        
        # 批量执行插入
        with conn:
            conn.executemany(insert_sql, data)
        
        total_rows += len(data)
        if i % 10 == 0:  # 每 10 批打印一次进度
            elapsed = time.time() - start_time
            speed = total_rows / elapsed
            print(f"Inserted {total_rows:,} rows, elapsed {elapsed:.1f}s, speed {speed:.0f} rows/s")
    
    # 创建索引（主键已自动创建，无需额外操作）
    # 可执行 VACUUM 以优化数据库大小
    conn.execute("VACUUM")
    conn.close()
    print(f"Done! Total {total_rows:,} rows inserted, database file: {target_file}")
    return
