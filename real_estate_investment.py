import numpy as np
from dataclasses import dataclass, field
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output
from flask import Flask

# Constants
HOME_PRICE = 1250000.0
DOWNPAYMENT_PERCENTAGE = 0.2
CLOSING_COST = 45000.0
PROPERTY_TAX_RATE = 0.0136
INTEREST_RATE = 0.06375
APPRECIATION_RATE = 0.065
INSURANCE = 2000.0  # per year
LOAN_TERM = 30
S_P_500_RETURN = 0.07
INITIAL_INCOME = 200000
INCOME_GROWTH_RATE = 0.03
INCOME_TAX_RATE = 0.4
EXPENSE_GROWTH_RATE = 0.06
INITIAL_RENT = 3000.0
RENT_GROWTH_RATE = 0.045
YEARLY_RSU_GRANT = 10000.0
PRE_TAX_401K_CONTRIBUTION = 23000.0
OPTIMUM_DOWNPAYMENT_PERCENTAGE = 0.3
# Personal expenses
INVESTMENT = 500.0
GROCERY = 300.0
UTILITY = 200.0
SUBSCRIPTION = 100.0
LUNCH = 200.0
LIFESTYLE = 100.0
MEDICAL = 100.0
TRAVEL = 200.0
SHOPPING = 200.0
MAINTENANCE = 0.0
CAR = 500.0
MISCELLANEOUS = 100.0

@dataclass
class LoanDetails:
    house_price: float
    downpayment_percentage: float
    closing_cost: float
    interest_rate: float
    loan_amount: float = field(init=False)
    monthly_interest_rate: float = field(init=False)
    num_payments: int = field(init=False)
    downpayment: float = field(init=False)
    appreciation_rate: float = APPRECIATION_RATE

    def __post_init__(self):
        self.downpayment = self.house_price * self.downpayment_percentage
        self.loan_amount = self.house_price - self.downpayment
        self.monthly_interest_rate = self.interest_rate / 12
        self.num_payments = LOAN_TERM * 12

@dataclass
class HouseExpenses:
    loan_details: LoanDetails
    property_tax_rate: float = PROPERTY_TAX_RATE
    monthly_mortgage: float = field(init=False)
    property_tax: float = field(init=False)
    total_home_expenses: float = field(init=False)

    def __post_init__(self):
        self.calculate_expenses()

    def calculate_expenses(self):
        self.monthly_mortgage = self.calculate_monthly_mortgage()
        self.property_tax = (self.loan_details.house_price * self.property_tax_rate) / 12
        self.total_home_expenses = self.monthly_mortgage + self.property_tax + INSURANCE / 12

    def calculate_monthly_mortgage(self):
        L = self.loan_details.loan_amount
        r = self.loan_details.monthly_interest_rate
        n = self.loan_details.num_payments
        M = L * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        return M

@dataclass
class PersonalExpenses: # per month
    investment: float = INVESTMENT
    grocery: float = GROCERY
    utility: float = UTILITY
    subscription: float = SUBSCRIPTION
    lunch: float = LUNCH
    lifestyle: float = LIFESTYLE
    medical: float = MEDICAL
    travel: float = TRAVEL
    shopping: float = SHOPPING
    maintenance: float = MAINTENANCE
    car: float = CAR
    miscellaneous: float = MISCELLANEOUS

    def total_personal_expenses(self):
        return self.grocery + self.utility + self.subscription + self.lunch + self.lifestyle + self.medical + self.travel + self.shopping + self.maintenance + self.car + self.miscellaneous

    def total_personal_expenses_with_investment(self):
        return self.total_personal_expenses() + self.investment

@dataclass
class RentExpenses:
    initial_rent: float = INITIAL_RENT
    rent_growth_rate: float = RENT_GROWTH_RATE

    def annual_rent(self, year: int) -> float:
        return self.initial_rent * 12 * ((1 + self.rent_growth_rate) ** (year - 1))

@dataclass
class IncomeDetails:
    initial_income: float = INITIAL_INCOME
    growth_rate: float = INCOME_GROWTH_RATE
    tax_rate: float = INCOME_TAX_RATE
    pre_tax_401k_contribution: float = PRE_TAX_401K_CONTRIBUTION

    def annual_income(self, year: int) -> float:
        gross_income = self.initial_income * ((1 + self.growth_rate) ** (year - 1))
        taxable_income = gross_income - self.pre_tax_401k_contribution
        net_income = taxable_income * (1 - self.tax_rate)
        return net_income

@dataclass
class InvestmentGrowth:
    monthly_investment: float
    yearly_rsu_grant: float
    pre_tax_401k_contribution: float
    investment_growth_rate: float = S_P_500_RETURN

    def accumulated_value(self, years: int) -> float:
        total_value = 0.0
        for year in range(1, years + 1):
            total_value += self.yearly_rsu_grant
            total_value *= (1 + self.investment_growth_rate)
            total_value += self.monthly_investment * 12 * ((1 + self.investment_growth_rate) ** year)
        return total_value

    def accumulated_401k_value(self, years: int) -> float:
        total_401k_value = 0.0
        for year in range(1, years + 1):
            total_401k_value += self.pre_tax_401k_contribution
            total_401k_value *= (1 + self.investment_growth_rate)
        return total_401k_value

class RealEstateInvestment:
    def __init__(self, loan_details: LoanDetails, house_expenses: HouseExpenses, personal_expenses: PersonalExpenses, rent_expenses: RentExpenses, income_details: IncomeDetails, investment_growth: InvestmentGrowth):
        self.loan_details = loan_details
        self.house_expenses = house_expenses
        self.personal_expenses = personal_expenses
        self.rent_expenses = rent_expenses
        self.income_details = income_details
        self.investment_growth = investment_growth

    def calculate_payment_schedule(self):
        L = self.loan_details.loan_amount
        r = self.loan_details.monthly_interest_rate
        n = self.loan_details.num_payments
        M = self.house_expenses.monthly_mortgage

        schedule = np.zeros((n, 3))  # columns: total payment, principal payment, interest payment

        for i in range(n):
            interest_payment = L * r
            principal_payment = M - interest_payment
            L -= principal_payment

            schedule[i, 0] = M
            schedule[i, 1] = principal_payment
            schedule[i, 2] = interest_payment

        cumulative_schedule = np.cumsum(schedule, axis=0)
        return cumulative_schedule

    def investment_vs_house_value(self, schedule):
        years = np.arange(1, LOAN_TERM + 1)
        house_value = self.loan_details.house_price * ((1 + self.loan_details.appreciation_rate) ** years)
        total_investment = self.loan_details.downpayment + self.loan_details.closing_cost + np.cumsum([self.house_expenses.total_home_expenses * 12 for _ in years])
        sp500_investment = (self.loan_details.downpayment + self.loan_details.closing_cost) * ((1 + self.investment_growth.investment_growth_rate) ** years)
        loan_balance = self.loan_details.loan_amount - schedule[:years.size * 12:12, 1]
        roi = house_value - total_investment - loan_balance

        crossover_index_sp500 = np.argwhere(roi >= sp500_investment)
        crossover_year_sp500 = crossover_index_sp500.flatten()[0] + 1 if crossover_index_sp500.size > 0 else None

        crossover_index_breakeven = np.argwhere(roi >= 0)
        crossover_year_breakeven = crossover_index_breakeven.flatten()[0] + 1 if crossover_index_breakeven.size > 0 else None

        return years, house_value, total_investment, sp500_investment, roi, crossover_year_sp500, crossover_year_breakeven

    def optimal_time_to_buy(self):
        years = np.arange(1, LOAN_TERM + 1)
        current_investment = self.loan_details.downpayment + self.loan_details.closing_cost
        current_house_price = self.loan_details.house_price
        investment_over_time = []
        downpayment_30_over_time = []
        house_price_over_time = []

        for year in range(1, LOAN_TERM + 1):
            current_investment *= (1 + self.investment_growth.investment_growth_rate)
            current_house_price *= (1 + APPRECIATION_RATE)
            investment_over_time.append(current_investment)
            downpayment_30_over_time.append(current_house_price * OPTIMUM_DOWNPAYMENT_PERCENTAGE)
            house_price_over_time.append(current_house_price)

        optimal_year = np.argwhere(np.array(investment_over_time) >= np.array(downpayment_30_over_time)).flatten()[0] + 1

        return years, investment_over_time, downpayment_30_over_time, house_price_over_time, optimal_year

    def calculate_profit_and_loan_balance(self, schedule):
        years = np.arange(1, LOAN_TERM + 1)
        house_value = self.loan_details.house_price * ((1 + self.loan_details.appreciation_rate) ** years)
        total_investment = self.loan_details.downpayment + self.loan_details.closing_cost + np.cumsum([self.house_expenses.total_home_expenses * 12 for _ in years])

        loan_balance = self.loan_details.loan_amount - schedule[:years.size * 12:12, 1]
        profit = house_value - total_investment - loan_balance

        return years, loan_balance, profit

    def money_in_hand(self):
        years = np.arange(1, LOAN_TERM + 1)
        monthly_income = np.array([(self.income_details.annual_income(year) + self.income_details.pre_tax_401k_contribution) / 12 for year in years])
        monthly_personal_expenses = np.array([self.personal_expenses.total_personal_expenses() * ((1 + EXPENSE_GROWTH_RATE) ** (year - 1)) for year in years]) + self.personal_expenses.investment
        monthly_expenses_with_house = monthly_personal_expenses + self.house_expenses.total_home_expenses
        monthly_expenses_with_rent = monthly_personal_expenses + np.array([self.rent_expenses.annual_rent(year) / 12 for year in years])

        money_in_hand_with_house = monthly_income - monthly_expenses_with_house
        money_in_hand_with_rent = monthly_income - monthly_expenses_with_rent

        return years, money_in_hand_with_house, money_in_hand_with_rent

    def net_worth(self, schedule):
        years = np.arange(1, LOAN_TERM + 1)
        assets = self.loan_details.house_price * ((1 + self.loan_details.appreciation_rate) ** years) - (self.loan_details.loan_amount - schedule[:years.size * 12:12, 1])
        stocks = np.array([self.investment_growth.accumulated_value(year) for year in years])
        k401 = np.array([self.investment_growth.accumulated_401k_value(year) for year in years])
        total_assets = assets + stocks + k401
        return years, assets, stocks, total_assets, k401

    def create_plot(self, title, x_data, y_data, x_label, y_label, line_names, line_colors, vline=None):
        fig = go.Figure()
        for y, name, color in zip(y_data, line_names, line_colors):
            fig.add_trace(go.Scatter(x=x_data, y=y, mode='lines', name=name, line=dict(width=4, color=color)))
        if vline:
            for vline_info in vline:
                fig.add_vline(x=vline_info['x'], line=dict(color=vline_info['color'], width=2, dash='dash'), annotation_text=vline_info['text'])
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template='plotly_white', height=700, width=1200, font=dict(size=20))
        return fig

    def create_plots(self):
        schedule = self.calculate_payment_schedule()
        years, house_value, total_investment, sp500_investment, roi, crossover_year_sp500, crossover_year_breakeven = self.investment_vs_house_value(schedule)
        optimal_years, investment_over_time, downpayment_30_over_time, house_price_over_time, optimal_year = self.optimal_time_to_buy()
        years_profit, loan_balance, profit = self.calculate_profit_and_loan_balance(schedule)
        years_money, money_in_hand_with_house, money_in_hand_with_rent = self.money_in_hand()
        years_net_worth, assets, stocks, total_assets, k401 = self.net_worth(schedule)

        # Create all plots using the create_plot method
        vline = []
        if crossover_year_sp500:
            vline.append({'x': crossover_year_sp500, 'color': 'Red', 'text': f'Crossover with S&P 500 at Year {crossover_year_sp500}'})
        if crossover_year_breakeven:
            vline.append({'x': crossover_year_breakeven, 'color': 'Green', 'text': f'Break-even Point at Year {crossover_year_breakeven}'})
        fig1 = self.create_plot("Investment vs House Value", years, 
                                [house_value, total_investment, sp500_investment, roi], 
                                "Years", "Value ($)", 
                                ["House Value", "Total Investment", "S&P 500 Investment", "Return on Investment"], 
                                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
                                vline)

        vline = [{'x': optimal_year, 'color': 'Red', 'text': f'Optimal Year to Buy: {optimal_year}'}] if optimal_year else None
        fig2 = self.create_plot("Optimal Time to Buy", years, 
                                [investment_over_time, downpayment_30_over_time, house_price_over_time], 
                                "Years", "Investment Value ($)", 
                                ["Investment Over Time", f"{OPTIMUM_DOWNPAYMENT_PERCENTAGE * 100}% Downpayment", "House Price Over Time"], 
                                ['#1f77b4', '#ff7f0e', '#2ca02c'], 
                                vline)

        fig3 = self.create_plot("Profit and Loan Balance Over Time", years, 
                                [profit, loan_balance], 
                                "Years", "Profit and Loan Balance ($)", 
                                ["Profit", "Loan Balance"], 
                                ['#1f77b4', '#ff7f0e'])

        fig4 = self.create_plot("Payment Schedule", years, 
                                [schedule[:years.size * 12:12, 0], schedule[:years.size * 12:12, 1], schedule[:years.size * 12:12, 2]], 
                                "Years", "Cumulative Payment ($)", 
                                ["Cumulative Total Payment", "Cumulative Principal Payment", "Cumulative Interest Payment"], 
                                ['#1f77b4', '#ff7f0e', '#2ca02c'])

        fig5 = self.create_plot("Monthly Money in Hand", years, 
                                [money_in_hand_with_house, money_in_hand_with_rent], 
                                "Years", "Monthly Money in Hand ($)", 
                                ["Monthly Money in Hand with House", "Monthly Money in Hand with Rent"], 
                                ['#1f77b4', '#ff7f0e'])

        fig6 = self.create_plot("Net Worth", years_net_worth, 
                                [assets, stocks, total_assets, k401], 
                                "Years", "Net Worth ($)", 
                                ["Assets", "Stocks", "Total Assets", "401k Value"], 
                                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        return fig1, fig2, fig3, fig4, fig5, fig6

# Create Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

loan_details = LoanDetails(house_price=HOME_PRICE, downpayment_percentage=DOWNPAYMENT_PERCENTAGE, closing_cost=CLOSING_COST, interest_rate=INTEREST_RATE)
house_expenses = HouseExpenses(loan_details, PROPERTY_TAX_RATE)
personal_expenses = PersonalExpenses()
rent_expenses = RentExpenses()
income_details = IncomeDetails()
investment_growth = InvestmentGrowth(monthly_investment=personal_expenses.investment, yearly_rsu_grant=YEARLY_RSU_GRANT, pre_tax_401k_contribution=PRE_TAX_401K_CONTRIBUTION, investment_growth_rate=S_P_500_RETURN)
real_estate_investment = RealEstateInvestment(loan_details, house_expenses, personal_expenses, rent_expenses, income_details, investment_growth)

# Create plots
fig1, fig2, fig3, fig4, fig5, fig6 = real_estate_investment.create_plots()

# Define app layout
app.layout = html.Div(children=[
    html.H1(children='Real Estate Investment Analysis Dashboard', style={'font-size': '36px'}),

    html.Div([
        html.Div([
            html.Label('Home Price'),
            dcc.Input(id='home_price', type='number', value=HOME_PRICE, step=10000)
        ], className='input-box'),
        html.Div([
            html.Label('Downpayment Percentage'),
            dcc.Input(id='downpayment_percentage', type='number', value=DOWNPAYMENT_PERCENTAGE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Closing Cost'),
            dcc.Input(id='closing_cost', type='number', value=CLOSING_COST, step=1000)
        ], className='input-box'),
        html.Div([
            html.Label('Interest Rate'),
            dcc.Input(id='interest_rate', type='number', value=INTEREST_RATE, step=0.001)
        ], className='input-box'),
        html.Div([
            html.Label('Property Tax Rate'),
            dcc.Input(id='property_tax_rate', type='number', value=PROPERTY_TAX_RATE, step=0.0001)
        ], className='input-box'),
        html.Div([
            html.Label('House Appreciation Rate'),
            dcc.Input(id='appreciation_rate', type='number', value=APPRECIATION_RATE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Insurance'),
            dcc.Input(id='insurance', type='number', value=INSURANCE, step=100)
        ], className='input-box'),
        html.Div([
            html.Label('Household Income'),
            dcc.Input(id='initial_income', type='number', value=INITIAL_INCOME, step=10000)
        ], className='input-box'),
        html.Div([
            html.Label('Income Growth Rate'),
            dcc.Input(id='income_growth_rate', type='number', value=INCOME_GROWTH_RATE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Income Tax Rate'),
            dcc.Input(id='income_tax_rate', type='number', value=INCOME_TAX_RATE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Expense Growth Rate'),
            dcc.Input(id='expense_growth_rate', type='number', value=EXPENSE_GROWTH_RATE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Monthly Rent'),
            dcc.Input(id='initial_rent', type='number', value=INITIAL_RENT, step=100)
        ], className='input-box'),
        html.Div([
            html.Label('Rent Growth Rate'),
            dcc.Input(id='rent_growth_rate', type='number', value=RENT_GROWTH_RATE, step=0.01)
        ], className='input-box'),
        html.Div([
            html.Label('Yearly RSU Grant'),
            dcc.Input(id='yearly_rsu_grant', type='number', value=YEARLY_RSU_GRANT, step=1000)
        ], className='input-box'),
        html.Div([
            html.Label('Pre-tax 401k Contribution'),
            dcc.Input(id='pre_tax_401k_contribution', type='number', value=PRE_TAX_401K_CONTRIBUTION, step=1000)
        ], className='input-box'),
        html.Div([
            html.Label('Investment'),
            dcc.Input(id='investment', type='number', value=INVESTMENT, step=100)
        ], className='input-box'),
        html.Div([
            html.Label('S&P 500 Return'),
            dcc.Input(id='sp500_return', type='number', value=S_P_500_RETURN, step=0.001)
        ], className='input-box'),
        html.Div([
            html.Label('Grocery'),
            dcc.Input(id='grocery', type='number', value=GROCERY, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Utility'),
            dcc.Input(id='utility', type='number', value=UTILITY, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Subscription'),
            dcc.Input(id='subscription', type='number', value=SUBSCRIPTION, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Lunch'),
            dcc.Input(id='lunch', type='number', value=LUNCH, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Lifestyle'),
            dcc.Input(id='lifestyle', type='number', value=LIFESTYLE, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Medical'),
            dcc.Input(id='medical', type='number', value=MEDICAL, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Travel'),
            dcc.Input(id='travel', type='number', value=TRAVEL, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Shopping'),
            dcc.Input(id='shopping', type='number', value=SHOPPING, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Maintenance'),
            dcc.Input(id='maintenance', type='number', value=MAINTENANCE, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Car'),
            dcc.Input(id='car', type='number', value=CAR, step=50)
        ], className='input-box'),
        html.Div([
            html.Label('Miscellaneous'),
            dcc.Input(id='miscellaneous', type='number', value=MISCELLANEOUS, step=50)
        ], className='input-box')
    ], className='input-container'),

    html.Button(id='update-button', n_clicks=0, children='Update', className='update-button'),

    dcc.Graph(
        id='investment-vs-house-value',
        figure=fig1
    ),

    dcc.Graph(
        id='optimal-time-to-buy',
        figure=fig2
    ),

    dcc.Graph(
        id='profit-and-loan-balance-over-time',
        figure=fig3
    ),

    dcc.Graph(
        id='payment-schedule',
        figure=fig4
    ),

    dcc.Graph(
        id='monthly-money-in-hand',
        figure=fig5
    ),

    dcc.Graph(
        id='net-worth',
        figure=fig6
    ),
], className='main-container')

@app.callback(
    [Output('investment-vs-house-value', 'figure'),
     Output('optimal-time-to-buy', 'figure'),
     Output('profit-and-loan-balance-over-time', 'figure'),
     Output('payment-schedule', 'figure'),
     Output('monthly-money-in-hand', 'figure'),
     Output('net-worth', 'figure')],
    [Input('update-button', 'n_clicks')],
    [Input('home_price', 'value'),
     Input('downpayment_percentage', 'value'),
     Input('closing_cost', 'value'),
     Input('interest_rate', 'value'),
     Input('property_tax_rate', 'value'),
     Input('appreciation_rate', 'value'),
     Input('insurance', 'value'),
     Input('initial_income', 'value'),
     Input('income_growth_rate', 'value'),
     Input('income_tax_rate', 'value'),
     Input('expense_growth_rate', 'value'),
     Input('initial_rent', 'value'),
     Input('rent_growth_rate', 'value'),
     Input('yearly_rsu_grant', 'value'),
     Input('pre_tax_401k_contribution', 'value'),
     Input('investment', 'value'),
     Input('sp500_return', 'value'),
     Input('grocery', 'value'),
     Input('utility', 'value'),
     Input('subscription', 'value'),
     Input('lunch', 'value'),
     Input('lifestyle', 'value'),
     Input('medical', 'value'),
     Input('travel', 'value'),
     Input('shopping', 'value'),
     Input('maintenance', 'value'),
     Input('car', 'value'),
     Input('miscellaneous', 'value')]
)
def update_plots(n_clicks, home_price, downpayment_percentage, closing_cost, interest_rate, property_tax_rate, appreciation_rate, insurance, initial_income, income_growth_rate, income_tax_rate, expense_growth_rate, initial_rent, rent_growth_rate, yearly_rsu_grant, pre_tax_401k_contribution, investment, sp500_return, grocery, utility, subscription, lunch, lifestyle, medical, travel, shopping, maintenance, car, miscellaneous):
    # Update constants with default values if None
    home_price = home_price or HOME_PRICE
    downpayment_percentage = downpayment_percentage or DOWNPAYMENT_PERCENTAGE
    closing_cost = closing_cost or CLOSING_COST
    interest_rate = interest_rate or INTEREST_RATE
    property_tax_rate = property_tax_rate or PROPERTY_TAX_RATE
    appreciation_rate = appreciation_rate or APPRECIATION_RATE
    insurance = insurance or INSURANCE
    initial_income = initial_income or INITIAL_INCOME
    income_growth_rate = income_growth_rate or INCOME_GROWTH_RATE
    income_tax_rate = income_tax_rate or INCOME_TAX_RATE
    expense_growth_rate = expense_growth_rate or EXPENSE_GROWTH_RATE
    initial_rent = initial_rent or INITIAL_RENT
    rent_growth_rate = rent_growth_rate or RENT_GROWTH_RATE
    yearly_rsu_grant = yearly_rsu_grant or YEARLY_RSU_GRANT
    pre_tax_401k_contribution = pre_tax_401k_contribution or PRE_TAX_401K_CONTRIBUTION
    investment = investment or INVESTMENT
    sp500_return = sp500_return or S_P_500_RETURN
    grocery = grocery or GROCERY
    utility = utility or UTILITY
    subscription = subscription or SUBSCRIPTION
    lunch = lunch or LUNCH
    lifestyle = lifestyle or LIFESTYLE
    medical = medical or MEDICAL
    travel = travel or TRAVEL
    shopping = shopping or SHOPPING
    maintenance = maintenance or MAINTENANCE
    car = car or CAR
    miscellaneous = miscellaneous or MISCELLANEOUS

    # Recalculate everything with new constants
    loan_details = LoanDetails(house_price=home_price, downpayment_percentage=downpayment_percentage, closing_cost=closing_cost, interest_rate=interest_rate)
    house_expenses = HouseExpenses(loan_details, property_tax_rate)
    personal_expenses = PersonalExpenses(investment=investment, grocery=grocery, utility=utility, subscription=subscription, lunch=lunch, lifestyle=lifestyle, medical=medical, travel=travel, shopping=shopping, maintenance=maintenance, car=car, miscellaneous=miscellaneous)
    rent_expenses = RentExpenses(initial_rent=initial_rent, rent_growth_rate=rent_growth_rate)
    income_details = IncomeDetails(initial_income=initial_income, growth_rate=income_growth_rate, tax_rate=income_tax_rate, pre_tax_401k_contribution=pre_tax_401k_contribution)
    investment_growth = InvestmentGrowth(monthly_investment=personal_expenses.investment, yearly_rsu_grant=yearly_rsu_grant, pre_tax_401k_contribution=pre_tax_401k_contribution, investment_growth_rate=sp500_return)
    real_estate_investment = RealEstateInvestment(loan_details, house_expenses, personal_expenses, rent_expenses, income_details, investment_growth)

    # Recalculate house expenses to reflect updated loan details
    house_expenses.calculate_expenses()

    # Create updated plots
    fig1, fig2, fig3, fig4, fig5, fig6 = real_estate_investment.create_plots()

    return fig1, fig2, fig3, fig4, fig5, fig6

# Add CSS styling for input boxes and container
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Real Estate Investment Analysis Dashboard</title>
        <style>
            .main-container {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .input-container {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 5px;
                margin-bottom: 30px;
            }
            .input-box {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 10px;
                background-color: #f0f0f0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                font-size: 16px;
            }
            .update-button {
                margin: 20px 0;
                padding: 15px 30px;
                font-size: 18px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            .update-button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app on a specific IP address and port
if __name__ == '__main__':
    app.run_server(debug=True, host='192.168.1.131', port=8080)
