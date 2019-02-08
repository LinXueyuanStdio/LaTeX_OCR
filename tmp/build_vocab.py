import click


from model.utils.data_generator import DataGenerator
from model.utils.text import build_vocab, write_vocab
from model.utils.image import build_images
from model.utils.general import Config


@click.command()
@click.option('--data', default="configs/data.json", help='Path to data json config')
@click.option('--vocab', default="configs/vocab.json", help='Path to vocab json config')
def main(data, vocab):
    data_config = Config(data)

    # datasets
    train_set = DataGenerator(path_formulas=data_config.path_formulas_train,
                              dir_images=data_config.dir_images_train,
                              path_matching=data_config.path_matching_train)
    # vocab
    vocab_config = Config(vocab)
    vocab = build_vocab([train_set], min_count=vocab_config.min_count_tok)
    write_vocab(vocab, vocab_config.path_vocab)


if __name__ == "__main__":
    main()
