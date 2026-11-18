'''
Registry of experiment stages.

Maps each stage name (the ``run``, ``summarize`` and ``status`` argument, and the results
sub-directory) to the module that implements it. The runner imports these modules lazily, so a
stage that is not written yet, or fails to import, only affects runs that name it.
'''

REGISTRY: dict[str, str] = {
    'smoke': 'lplearn.experiments.stages.smoke',
    'lp_solutions': 'lplearn.experiments.stages.lp_solutions',
    'learners': 'lplearn.experiments.stages.learners',
    'bound': 'lplearn.experiments.stages.bound',
    'finite_n': 'lplearn.experiments.stages.finite_n',
    'commons': 'lplearn.experiments.stages.commons',
    'figures': 'lplearn.experiments.stages.figures',
    'tables': 'lplearn.experiments.stages.tables',
    'tutorial': 'lplearn.experiments.stages.tutorial',
    'web': 'lplearn.experiments.stages.web',
}
