class Baseline:
    def __init__(self, model, decoder_configs):
        self.model = model
        self.decoder_configs = decoder_configs

    def forward(self, input):
        logits = self.model.generate_logits(input)
        decoded = self.model.decode(logits)
        return decoded
