from django.core.management.base import BaseCommand
from bandit.training import BanditTrainer


class Command(BaseCommand):
    help = 'Train a new bandit model from scratch'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Number of training epochs (default: 50)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training (default: 32)'
        )

    def handle(self, *args, **options):
        epochs = options['epochs']
        batch_size = options['batch_size']
        
        self.stdout.write(self.style.SUCCESS(
            f'\nTraining new model with:\n'
            f'  Epochs: {epochs}\n'
            f'  Batch size: {batch_size}\n'
        ))
        
        trainer = BanditTrainer()
        
        try:
            history = trainer.train_new_model(
                epochs=epochs,
                batch_size=batch_size,
            )
            
            self.stdout.write(self.style.SUCCESS(
                f'\n✅ Training completed successfully!\n'
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Training failed: {str(e)}'))
            raise