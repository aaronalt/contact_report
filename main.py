import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sidetable
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

total_contacted = 0

def construct_v_graph(df):
    x_labels = [label for label in df[list(df)[0]]]
    count = [c for c in df[list(df)[1]]]
    percent = [round(p, 3) for p in df[list(df)[2]]]
    x = np.arange(len(x_labels))
    width = 0.25
    fig, ax = plt.subplots()
    bar1 = ax.bar(x + width, count, width, label="Sum of contacts")
    # bar2 = ax.bar(x + width/2, percent, width, label="% of contacts")
    ax.set_ylabel('Count')
    ax.set_title(f'Grouped by {list(df.columns)[0]}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    graph = plt.savefig('vertical_graph.png')
    return graph


def construct_h_graph(df):
    countries = [label for label in df[list(df)[0]]]
    count = [c for c in df[list(df)[1]]]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(countries))
    ax.barh(y_pos, count)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries)
    ax.invert_yaxis()
    ax.set_xlabel('Contact Count')
    ax.set_title("New Contacts Per Count")
    fig.tight_layout()
    fig.set_size_inches(6, 7)
    graph = plt.savefig("horizontal_graph.png")
    return graph


def construct_hhh_graph(df):
    countries = []
    results = []
    counts = []
    for country, dic in master.items():
        countries.append(country)
        for result, count in dic.items():
            results.append(result)
            counts.append(count)
    fig, ax = plt.subplots()
    y = np.arange(len(countries))
    width = 0.35
    ax.barh(y - width/2, results, width, label='Results')
    ax.barh()
    ax.set_yticks(y)
    ax.set_yticklabels(countries)
    ax.invert_yaxis()
    ax.set_xlabel('Contact Count')
    ax.set_title("New Contacts Per Count")
    fig.tight_layout()
    fig.set_size_inches(6, 7)
    graph = plt.savefig("horizontal_graph.png")
    return


def freq_of_result(df):
    st = df.stb.freq(['result'])
    st_f = fix_percentage(st)
    return st_f


def freq_of_country(df, for_graph=False):
    st = df.stb.freq(['country'])
    if for_graph:
        return st
    st_f = fix_percentage(st)
    return st_f


master = {}


def count_by_result(df):
    pt = df.pivot_table(values=['status'], index=['country', 'result'], aggfunc=[np.count_nonzero])
    global master, total_contacted
    for idx, row in pt.iterrows():
        row = str(row).replace('\n', ' ')
        count = str(row).split(" ")[6]
        country = idx[0]
        result = idx[1]
        total_contacted += int(count)
        if country not in master:
            master[country] = {result: count}
        else:
            master[country].update({result: count})
    # pt.reset_index()
    # plot = pt.plot(kind="bar")

    return pt


def count_by_response(df):
    pt = df[df.result == 'Response'].pivot_table(values=['result'], index=['country'], aggfunc=[np.count_nonzero])
    response_dict = {}
    for i in pt.iterrows():
        response_dict[i[0]] = i[1][0]
    for country, ccount in count_contacts_by_country(df).items():
        try:
            if response_dict[country]:
                response_dict[country] = {"Contacted": ccount, "Responses": response_dict[country]}
        except KeyError as e:
            pass
    print(response_dict)
    pt['Contacted count'] = [j['Contacted'] for i, j in response_dict.items()]
    resp_rate = []
    resp_rate_to_total = []
    for i, j in response_dict.items():
        y = j['Contacted']
        x = j['Responses']
        z = str(round(100*int(x) / int(y), 2)) + '%'
        resp_rate.append(z)
        w = str(round(100*int(x) / total_contacted, 2)) + '%'
        resp_rate_to_total.append(w)
    pt['Response rate contact'] = resp_rate
    pt['Response rate total'] = resp_rate_to_total
    # todo: sort values in ascending order
    return pt, response_dict


def count_contacts_by_country(df):
    ccount = {}
    for i in freq_of_country(df).iterrows():
        ccount[i[1][0]] = i[1][1]
    return ccount


def fix_percentage(st):
    percent_fix = (100. * st.Percent / st.Percent.sum()).round(2).astype(str) + '%'
    st.Percent = percent_fix
    return st


def read_df():
    pd.options.display.max_columns = None
    df = pd.read_csv('New Contacts-Grid view.csv')
    return df


def render_html(dfs, plot=None):
    html = Environment(
        loader=FileSystemLoader(searchpath='')
    ).get_template('contact_report.html').render(dfs=dfs, plot=plot)
    with open('new_contacts_report.pdf', "w+b") as out_pdf_file:
        pisa.CreatePDF(src=html, dest=out_pdf_file)


def main():
    df = read_df()
    by_result_count = freq_of_result(df)
    by_countries = freq_of_country(df)
    by_result_per_country = count_by_result(df)
    by_response, response_dict = count_by_response(df)
    """print(by_result_count)
    # print(construct_graph(by_results))
    print(by_countries.head())
    # print(construct_graph(by_countries))
    print(by_result_per_country.head())
    print(count_by_response(df)[0])"""

    print(by_result_per_country)
    #g = construct_hhh_graph(by_result_per_country)
    #g.show()
    #print(g)

    # render to HTML
    render_dict = {'by_result_count': by_result_count,
                   'by_response_rate': count_by_response(df)[0],
                   'by_countries': by_countries,
                   'by_result_per_country': by_result_per_country,
                   }
    render_html(render_dict, construct_h_graph(freq_of_country(df, for_graph=True)))


if __name__ == '__main__':
    main()
