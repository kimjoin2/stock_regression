import glob


def run():
    csv_list = glob.glob('stock_data/*')
    train_save_file = open('preprocessed_data/train_data.csv', 'w')
    for target_file in csv_list:
        pre_stock_data = []
        max_stock_high = None
        max_stock_low = None
        max_stock_open = None
        max_stock_close = None
        max_stock_volume = None
        min_stock_high = None
        min_stock_low = None
        min_stock_open = None
        min_stock_close = None
        min_stock_volume = None
        data_file = open(target_file)
        data_lines = data_file.readlines()[1:]
        data_file.close()
        if len(data_lines) < 365:
            continue

        for index, line in enumerate(data_lines):
            line_split = line.strip().split(',')
            if index != 0:
                next_stock_close = float(line_split[4].strip())
                price_up_rate = 100 * (next_stock_close - stock_close_for_check) / stock_close_for_check
                if abs(price_up_rate) <= 30:
                    pre_stock_data.append((stock_high, stock_low, stock_open, stock_close, stock_volume, price_up_rate))
            stock_open = float(line_split[3].strip())
            if stock_open == 0:
                continue
            stock_high = (float(line_split[1].strip()) - stock_open) / stock_open
            stock_low = (stock_open - float(line_split[2].strip())) / stock_open
            stock_close = (float(line_split[4].strip()) - stock_open) / stock_open
            stock_close_for_check = float(line_split[4].strip())
            stock_volume = float(line_split[5].strip()) * (stock_open + stock_close)/2
            if len(pre_stock_data) == 0:
                min_stock_high = max_stock_high = stock_high
                min_stock_low = max_stock_low = stock_low
                min_stock_open = max_stock_open = stock_open
                min_stock_close = max_stock_close = stock_close
                min_stock_volume = max_stock_volume = stock_volume
            else:
                max_stock_high = max(stock_high, max_stock_high)
                max_stock_low = max(stock_low, max_stock_low)
                max_stock_open = max(stock_open, max_stock_open)
                max_stock_close = max(stock_close, max_stock_close)
                max_stock_volume = max(stock_volume, max_stock_volume)

                min_stock_high = min(stock_high, min_stock_high)
                min_stock_low = min(stock_low, min_stock_low)
                min_stock_open = min(stock_open, min_stock_open)
                min_stock_close = min(stock_close, min_stock_close)
                min_stock_volume = min(stock_volume, min_stock_volume)

        diff_high = max_stock_high - min_stock_high
        diff_low = max_stock_low - min_stock_low
        diff_open = max_stock_open - min_stock_open
        diff_close = max_stock_close - min_stock_close
        diff_volume = max_stock_volume - min_stock_volume
        for (stock_high, stock_low, stock_open, stock_close, stock_volume, price_up) in pre_stock_data:
            data_high = (stock_high - min_stock_high) / diff_high
            data_low = (stock_low - min_stock_low) / diff_low
            data_open = (stock_open - min_stock_open) / diff_open
            data_close = (stock_close - min_stock_close) / diff_close
            data_volume = (stock_volume - min_stock_volume) / diff_volume
            data_price_up = price_up
            train_save_file.write(str(data_high) +
                                  ',' + str(data_low) +
                                  ',' + str(data_open) +
                                  ',' + str(data_close) +
                                  ',' + str(data_volume) +
                                  ',' + str(data_price_up) + '\n')


    # parameter_file = open('preprocessed_data/parameter.csv', 'a')
    # parameter_file.write(
    #     str(min_stock_high) + ',' + str(diff_high) + ',' +
    #     str(min_stock_low) + ',' + str(diff_low) + ',' +
    #     str(min_stock_open) + ',' + str(diff_open) + ',' +
    #     str(min_stock_close) + ',' + str(diff_close) + ',' +
    #     str(min_stock_volume) + ',' + str(diff_volume)
    # )
    train_save_file.close()
    print('done')


run()
