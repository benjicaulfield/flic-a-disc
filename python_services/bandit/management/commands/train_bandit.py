from django.core.management.base import BaseCommand
from bandit.training import BanditTrainer


class Command(BaseCommand):
    help = 'Train a new bandit model from scratch'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs (default: 50)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training (default: 32)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)'
        )

    def handle(self, *args, **options):
        epochs = options['epochs']
        batch_size = options['batch_size']
        learning_rate = options['learning_rate']
        
        self.stdout.write(self.style.SUCCESS(
            f'\nTraining new model with:\n'
            f'  Epochs: {epochs}\n'
            f'  Batch size: {batch_size}\n'
            f'  Learning rate: {learning_rate}\n'
        ))
        
        trainer = BanditTrainer()
        
        try:
            history = trainer.train_new_model(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            self.stdout.write(self.style.SUCCESS(
                f'\n✅ Training completed successfully!\n'
                f'Final accuracy: {history["val_accuracy"][-1]:.2%}\n'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Training failed: {str(e)}'))
            raise