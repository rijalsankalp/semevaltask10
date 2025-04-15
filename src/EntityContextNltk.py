from src.NltkPipe import NltkPipe

class EntityContextExtractor(NltkPipe):

    def __init__(self, stanford_server_url='http://localhost:9000'):
        super().__init__(stanford_server_url)

    def extract_entity_contexts(self, text, min_context_len=20):
        """
        Extract sentences around each entity after coreference resolution.
        Returns a dictionary of entity -> context string.
        """
        resolved_sentences = self._resolve_coreferences(text)
        entities = self.get_entities(text)

        entity_contexts = {}

        for entity in entities:
            sentences = [s for s in resolved_sentences if entity in s]
            if not sentences:
                continue

            context = " ".join(sentences)
            if len(context) < min_context_len:
                continue

            entity_contexts[entity] = context

        return entity_contexts
