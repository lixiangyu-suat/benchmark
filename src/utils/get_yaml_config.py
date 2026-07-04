
def yaml_config():
    """
    打造一个纯粹的底层工具函数, 这个函数像一个纯粹的加工厂，只负责“给路径 -> 吐字典”，不管路径是怎么来的。
    单一职责原则!
    """
    sys_args = parser.parse_args()

    # 读取 YAML 为一个 Python 字典
    with open(sys_args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)