'''
This module contains color palettes for the visualizations.
'''


OKABE_ITO = {  # idx: 0
    'Black': '#000000',
    'Orange': '#E69F00',
    'Sky Blue': '#56B4E9',
    'Bluish Green': '#009E73',
    'Yellow': '#F0E442',
    'Blue': '#0072B2',
    'Vermilion': '#D55E00',
    'Reddish Purple': '#CC79A7'
}


PALETTE_A = {  # idx: 1
    'Light Green': '#27b376',
    'Dark Green': '#006f3c',
    'Orange': '#f9a73e',
    'Red': '#bf212f',
}


LIGHTS = {  # idx: 2 -- OK
    'Dark Green': '#007000',
    'Light Green': '#238823',
    'Yellow': '#FFBF00',
    'Red': '#D2222D',
}


VINTAGE = {  # idx: 3
    'Dark Green': '#507B58',
    'Light Green': '#799163',
    'Yellow': '#EDE0A6',
    'Light Red': '#AB3131',
    'Dark Red': '#540202',
}


SATURATED = {  # idx: 4  - ok
    'Dark Green': '#006B3D',
    'Light Green': '#069C56',
    'Yellow': '#FF980E',
    'Orange': '#FF681E',
    'Red': '#D3212C',
}


NEON = {  # idx: 5
    'Dark Green': '#44ce1b',
    'Light Green': '#bbdb44',
    'Yellow': '#f7e379',
    'Orange': '#f2a134',
    'Red': '#e51f1f',
}


OFUSCATED = {  # idx: 6
    'Dark Green': '#056517',
    'Green': '#3f8f29',
    'Light Green': '#759116',
    'Dark Red': '#bf1029',
    'Light Red': '#de1a24',
}


idx = 1  # 0: OKABE_ITO, 1: PALETTE_A, 2: LIGHTS, 3: VINTAGE, 4: SATURATED, 5: NEON, 6: OFUSCATED
DEFAULT = {
    'a0b0': [OKABE_ITO['Yellow'], PALETTE_A['Orange'], LIGHTS['Yellow'], VINTAGE['Light Red'], SATURATED['Yellow'], NEON['Orange'], OFUSCATED['Light Red']][idx],
    'a0b1': [OKABE_ITO['Orange'], PALETTE_A['Red'], LIGHTS['Red'], VINTAGE['Dark Red'], SATURATED['Orange'], NEON['Red'], OFUSCATED['Dark Red']][idx],
    'a1b0': [OKABE_ITO['Sky Blue'], PALETTE_A['Light Green'], LIGHTS['Light Green'], VINTAGE['Light Green'], SATURATED['Light Green'], NEON['Light Green'], OFUSCATED['Light Green']][idx],
    'a1b1': [OKABE_ITO['Blue'], PALETTE_A['Dark Green'], LIGHTS['Dark Green'], VINTAGE['Dark Green'], SATURATED['Dark Green'], NEON['Dark Green'], OFUSCATED['Dark Green']][idx],
}
