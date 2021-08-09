

### old versions of some functions and some functions that are not currently used


def real_draw_chart(workbook, worksheet, title, x_axis, y_axis, categories_coord, values_coord, chart_coord, data_series, colors):
    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth'})
    for plot in data_series:
        chart.add_series({
            'name': str(plot),
            'line':   {'width': 1.25, 'color': colors[plot]},
            'categories': categories_coord,
            'values': values_coord[data_series.index(plot)],
        })

    chart.set_title({'name': title})
    chart.set_x_axis(x_axis)
    chart.set_y_axis(y_axis)

    chart.set_style(15)
    worksheet.insert_chart(chart_coord, chart, {'x_offset': 50, 'y_offset': 50, 'x_scale': 1.5, 'y_scale': 1.5})


def real_write_to_excel(file_path, sensitivities, learning_sample):
    R_learning = []
    for illuminant_index in range(illuminants_number):
            R_learning += [R[patch % patches_number] for patch in learning_sample 
                        if illuminant_index * patches_number <= patch < illuminant_index * patches_number + patches_number]
    R_learning = np.transpose(np.array(R_learning))

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    workbook = writer.book
    pd.DataFrame(sensitivities).to_excel(writer, sheet_name='Sheet1',
                            index=False, header=channels, startrow=1, startcol=1)
    pd.DataFrame(R_learning).to_excel(writer, sheet_name='Sheet2',
                            index=False, header=learning_sample, startrow=1, startcol=1)
    worksheet_1 = writer.sheets['Sheet1']
    worksheet_2 = writer.sheets['Sheet2']
    bold = workbook.add_format({'bold': 1})

    for row_num, data in enumerate(wavelengths):
        worksheet_1.write(row_num + 2, 0, data)
        worksheet_2.write(row_num + 2, 0, data)
    worksheet_1.write('A2', 'wavelegths', bold)
    worksheet_1.write('C1', 'Sensitivities', bold)
    worksheet_2.write('A2', 'wavelegths', bold)
    worksheet_2.write('B1', "Patches' reflectances", bold)

    sensitivities_x_axis = {'name': 'Wavelengths, nm', 'min': wavelengths[0], 'max': wavelengths[-1]}
    sensitivities_y_axis = {'name': 'Sensitivities Function'}
    sensitivities_values_coord = []  
    for channel in range(3):
        value_letter = alphabet[alphabet.index('B') + channel]
        sensitivities_values_coord.append('=Sheet1!$' + value_letter + '$3:$' + value_letter + '$109') 
    draw_chart(workbook, worksheet_1, 'Sensitivities', sensitivities_x_axis, sensitivities_y_axis, \
        '=Sheet1!$A$3:$A$109', sensitivities_values_coord, 'F1', channels, colors_RGB)

    patches_x_axis = {'name': 'Wavelengths, nm', 'min': wavelengths[0], 'max': wavelengths[-1]}
    patches_y_axis = {'name': 'Reflectance spectra'}
    patches_values_coord = []  
    for patch in range(len(R_learning[0])):
        value_letter = alphabet[alphabet.index('B') + patch]
        patches_values_coord.append('=Sheet2!$' + value_letter + '$3:$' + value_letter + '$109')
    # cmap = plt.cm.get_cmap('viridis')
    # colors_patches = [cmap(i) for i in range(cmap.N)] 
    colors_patches = {i:'blue' for i in learning_sample}
    
    draw_chart(workbook, worksheet_1, "Patches' reflectance", patches_x_axis, patches_y_axis, \
        '=Sheet2!$A$3:$A$109', patches_values_coord, 'F26', learning_sample, colors_patches)
    
    workbook.close()


def choose_learning_sample(patches_number, choosed_patches_number, illuminants_number, valid, achromatic_single, ratio=0.8):
    learning_sample = {}
    for channel in range(3):
        chromatic_learning_sample = []
        achromatic = [i * patches_number + single for i in range(illuminants_number) for single in achromatic_single]
        all_chromatic_potential = [patch for patch in valid[channel] if patch not in achromatic]
        chromatic_learning_number = int(ratio * len(valid[channel]) - len(achromatic_single))

        for i in range(illuminants_number):
            potential = [patch for patch in all_chromatic_potential if i * patches_number <= patch < i * patches_number + patches_number]
            chromatic_learning_sample += sorted(random.sample(potential, k=chromatic_learning_number))

        learning_sample[channel] = [patch for patch in valid if patch in chromatic_learning_sample or patch in achromatic]
    return learning_sample 


def plot_pictures(C, learning_sample, sensitivities_df, P):
    sensitivities = np.zeros(shape=(len(wavelengths), 3))
    for channel in range(3):
        P_learning = np.array([P[channel][patch] for patch in learning_sample[channel]])
        sensitivities[:, channel] = inv((C[channel].T @ C[channel]).astype(float)) @ C[channel].T @ P_learning

    plot_sens(wavelengths, sensitivities, sensitivities_df, show=True)
    plot_spectra(C.T, show=True)


def check_accuracy_angles(patches_number, stimulus_predicted, stimulus_genuine):
    angles = []
    norms = []

    for i in range(patches_number):
        predicted_stimulus = stimulus_predicted[i]
        unit_predicted_stimulus = predicted_stimulus / np.linalg.norm(predicted_stimulus)
        genuine_stimulus = stimulus_genuine[i]
        print(predicted_stimulus, genuine_stimulus)
        unit_genuine_stimulus = genuine_stimulus / np.linalg.norm(genuine_stimulus)
        dot_product = np.dot(unit_predicted_stimulus, unit_genuine_stimulus)
        angles.append(np.arccos(dot_product) * 180 / 3.1415)
        norms.append(np.linalg.norm(predicted_stimulus - genuine_stimulus, 2))


    mean_angle = sum(angles) / patches_number
    variance_angles = statistics.variance(angles)
    angles_fig = sns.histplot(angles).get_figure()

    mean_norm = np.mean(norms)
    variance_norms = statistics.variance(norms)
    norms_fig = sns.histplot(norms).get_figure()

    return mean_angle, variance_angles, angles_fig, mean_norm, variance_norms, norms_fig



def check_stimuls_accuracy(P, variances):
    # P /= P.max()
    # for channel in range(3): 
    #     mean_stimul = np.mean(P[:, channel])
    #     variance_stimuls = statistics.variance(P[:, channel])
    #     print(channel, mean_stimul, variance_stimuls)
    #     sns.histplot(P[:, channel], kde=True).get_figure()
    #     plt.show()

    # P /= P.max()

    # for channel in range(3):
    #     for stimul in range(len(P)): 
    #         for exposure in range(6):
    #             print(f'p: {stimul}, ch: {channel}, exp: {exposure}, std(%): \
    #                 {variances[stimul, channel, exposure] / P[stimul, channel, exposure] * 100}')
    #         print()

    for channel in range(3):
        for stimul in range(len(P)): 
            print(f'p: {stimul}, ch: {channel}, std(%): \
                    {variances[stimul, channel] / P[stimul, channel] * 100}')

