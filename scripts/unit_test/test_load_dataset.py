


def main():
    parser = parse_opts()
    args = parser.parse_args()

    train_dataset = RobotDataset(args=args, state='train') 
    val_dataset = RobotDataset(args=args, state='val')

    print(train_dataset)
    print(len(train_dataset))

    print(val_dataset)  
    print(len(val_dataset))

    # lapa_dataset = LapaDataset(args)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    
    from core.config.base_opts import parse_opts
    from core.dataset import RobotDataset, LapaDataset

    main()