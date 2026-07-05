from hypothesis import settings

settings.register_profile("default", max_examples=500, deadline=None)
settings.register_profile("thorough", max_examples=4000, deadline=None)
settings.load_profile("default")
