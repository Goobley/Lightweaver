boilerPlateNonTemplate = '''\
#include <utility>
// NOTE(cmo): Machine-Generated!
template <typename ...Args>
inline auto dispatch_{FnName}_({ArgList}, Args&& ...args)
-> {ReturnType}
{{
{SwitchGenBlock}

    switch (dispatcher__)
    {{
{Cases}
    default:
    {{
        assert(false);
    }} break;
    }}
}}
'''

boilerPlateSimdTemplate = '''\
#include <utility>
// NOTE(cmo): Machine-Generated!
template <SimdType simd, typename ...Args>
inline auto dispatch_{FnName}_({ArgList}, Args&& ...args)
-> {ReturnType}
{{
{SwitchGenBlock}

    switch (dispatcher__)
    {{
{Cases}
    default:
    {{
        assert(false);
    }} break;
    }}
}}
'''

argName = ['first', 'second', 'third', 'fourth',
           'fifth', 'sixth', 'seventh', 'eighth',
           'ninth', 'tenth', 'eleventh', 'twelfth',
           'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth']

def bool_dispatch_template_args(numSpecs, case):
    args = []
    for i in range(numSpecs):
        if case & (1 << i):
            args.append('true')
        else:
            args.append('false')
        args.append(', ')
    return ''.join(args[:-1])


def create_textual_specialisation_switch_bools(numSpec : int, fnName : str):
    if numSpec < 2:
        raise ValueError('Dispatcher not necessary for fewer than two specialisations.')

    if numSpec > 16:
        raise ValueError('This many specialisations seems silly.')

    if fnName == 'dispatcher__':
        raise ValueError('The name dispatcher__ is used internally to this function')

    formatArgs = {}
    formatArgs['FnName'] = fnName
    formatArgs['ArgList'] = ''.join([x for i in range(numSpec) for x in ('bool ', argName[i], ', ')][:-1])
    formatArgs['ReturnType'] = f"decltype({fnName}<{(' '.join(['false,' for _ in range(numSpec)]))[:-1]}>(std::forward<Args>(args)...))"

    switchGenBuilder = [f'u32 dispatcher__ = {argName[0]};\n']
    for i in range(1, numSpec):
        switchGenBuilder.append(f'dispatcher__ += {argName[i]} << {i};\n')
    switchGenBlock = ''.join(switchGenBuilder)
    switchGenBlock = '    ' + switchGenBlock.replace('\n', '\n    ')
    formatArgs['SwitchGenBlock'] = switchGenBlock

    casesBuilder = []
    for case in range(1 << numSpec):
        casesBuilder.append(f'case {case}:\n')
        casesBuilder.append('{\n')
        casesBuilder.append(f'    return {fnName}<{bool_dispatch_template_args(numSpec, case)}>(std::forward<Args>(args)...);\n')
        casesBuilder.append('} break;\n')
    cases = ''.join(casesBuilder)
    cases = '    ' + cases.replace('\n', '\n    ')
    formatArgs['Cases'] = cases

    return boilerPlateNonTemplate.format(**formatArgs)

def create_textual_specialisation_switch_bools_simd(numSpec : int, fnName : str):
    if numSpec < 2:
        raise ValueError('Dispatcher not necessary for fewer than two specialisations.')

    if numSpec > 16:
        raise ValueError('This many specialisations seems silly.')

    if fnName == 'dispatcher__':
        raise ValueError('The name dispatcher__ is used internally to this function')

    formatArgs = {}
    formatArgs['FnName'] = fnName
    formatArgs['ArgList'] = ''.join([x for i in range(numSpec) for x in ('bool ', argName[i], ', ')][:-1])
    formatArgs['ReturnType'] = f"decltype({fnName}<simd, {(' '.join(['false,' for _ in range(numSpec)]))[:-1]}>(std::forward<Args>(args)...))"

    switchGenBuilder = [f'u32 dispatcher__ = {argName[0]};\n']
    for i in range(1, numSpec):
        switchGenBuilder.append(f'dispatcher__ += {argName[i]} << {i};\n')
    switchGenBlock = ''.join(switchGenBuilder)
    switchGenBlock = '    ' + switchGenBlock.replace('\n', '\n    ')
    formatArgs['SwitchGenBlock'] = switchGenBlock

    casesBuilder = []
    for case in range(1 << numSpec):
        casesBuilder.append(f'case {case}:\n')
        casesBuilder.append('{\n')
        casesBuilder.append(f'    return {fnName}<simd, {bool_dispatch_template_args(numSpec, case)}>(std::forward<Args>(args)...);\n')
        casesBuilder.append('} break;\n')
    cases = ''.join(casesBuilder)
    cases = '    ' + cases.replace('\n', '\n    ')
    formatArgs['Cases'] = cases

    return boilerPlateSimdTemplate.format(**formatArgs)


if __name__ == '__main__':
    functions = [('chi_eta_aux_accum', 4), ('intensity_core_opt', 4),
                 ('compute_full_operator_rates', 2)]

    for fnName, nSpec in functions:
        with open(f'Dispatch_{fnName}.ipp', 'w') as f:
            f.write(create_textual_specialisation_switch_bools_simd(nSpec, fnName))