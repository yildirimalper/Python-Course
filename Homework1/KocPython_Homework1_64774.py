#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random

rand_s = random.uniform(0.5, 1.5)
rand_mf = random.uniform(0.9, 1.2)
rand_b = random.uniform(0.9, 1.1)

class Stock:
    def __init__(self, price, symbol):
        self.price = price
        self.symbol = symbol
            
class MutualFund:
    def __init__(self, symbol):
        self.price = 1
        self.symbol = symbol
        
class Bond(Stock):
    def __init__(self, price, symbol, coupon):
        super().__init__(price, symbol)
        self.coupon = coupon   
        
class Portfolio:
    asset_type = 'portfolio'
    def __init__(self):
        self.cash = 0
        self.funds = {}
        self.stocks = {}
        self.bonds = {}
        self.transactions = []
        self.num_shares = 0
        self.percentage_shares = 0
        self.bond_shares = 0
    
    def addCash(self, new_cash):
        if new_cash <= 0:
            print("Error, you cannot add negative cash to the cash account.")
            return
        else:
            self.cash += new_cash
            self.transactions.append(f"${new_cash} added to the cash account.")
            return self.cash 
        
    def buyStock(self, num_shares, Stock):
        if self.cash < num_shares*Stock.price:
            print("Error, you don't have enough cash to buy this amount of stocks.")
            return
        self.num_shares += num_shares
        self.stocks[Stock.symbol] = num_shares 
        self.cash -= num_shares*Stock.price
        self.transactions.append(f"{num_shares} shares of {Stock.symbol} added to the Stocks account.")
        return self.stocks
    
    def buyMutualFund(self, percentage_share, MutualFund):
        if percentage_share > 100 or percentage_share < 0:
            print("Error, you cannot have negative shares or more than 100 percentage shares of the fund.")
            return
        self.percentage_shares += percentage_share
        self.funds[MutualFund.symbol] = percentage_share
        self.cash -= percentage_share*MutualFund.price
        self.transactions.append(f"{percentage_share}% shares of {MutualFund.symbol} added to the Mutual Fund account.")
        return self.funds
        
    def withdrawCash(self, withdrawn_cash):
        if self.cash < withdrawn_cash:
            print("Error, you don't have enough cash to withdraw this amount.")
            return
        self.cash -= withdrawn_cash
        self.transactions.append(f"${withdrawn_cash} withdrawn from the cash account.")
        return self.cash
    
    def sellStock(self, sold_shares, Stock):
        if sold_shares == self.stocks[Stock.symbol]:
            self.stocks.pop(Stock.symbol)
            selling_price = rand_s*Stock.price           
        elif sold_shares < self.stocks[Stock.symbol]:
            selling_price = rand_s*Stock.price          
        self.cash += selling_price*sold_shares
        self.num_shares -= sold_shares
        self.stocks[Stock.symbol] -= sold_shares
        self.transactions.append(f"{sold_shares} number of {Stock.symbol} shares withdrawn from the stock account.")
        return self.stocks

    def sellMutualFund(self, sold_mf, MutualFund):
        if sold_mf == self.funds[MutualFund.symbol]:
            self.funds.pop(MutualFund)
            fund_sell_price = rand_mf*MutualFund.price
        elif sold_mf < self.funds[MutualFund.symbol]:
            fund_sell_price = rand_mf*MutualFund.price       
        self.cash += fund_sell_price*sold_mf
        self.percentage_shares -= sold_mf
        self.funds[MutualFund.symbol] -= sold_mf
        self.transactions.append(f"{sold_mf} percentage of {MutualFund.symbol} shares withdrawn from the mutual fund account.")
        return self.funds
            
    def print_portfolio(self):
        print(
        f"""Here is your portfolio:
        You have {self.cash} $ cash, 
        {self.stocks} amount of stocks, 
        {self.funds} shares of mutual funds.
        {self.bonds} shares of bonds.""")
        
    def history(self):
        print(*self.transactions, sep="\n")
    
    def buyBond(self, bond_shares, Bond):
        if self.cash < bond_shares*Bond.price:
            print("Error, you don't have enough cash to buy this amount of bonds.")
            return
        self.bond_shares += bond_shares
        self.bonds[Bond.symbol] = self.bond_shares 
        self.cash -= bond_shares*Bond.price
        self.cash += bond_shares*Bond.coupon
        self.transactions.append(f"{bond_shares} shares of {Bond.symbol} added to the Bonds account. \nYou received ${Bond.coupon*bond_shares} as your first coupon payment.")
        return self.bonds
    
    def sellBond(self, sold_bond, Bond):
        if sold_bond == self.bonds[Bond.symbol]:
            self.bonds.pop(Bond.symbol)
            bond_sell_price = rand_b*Bond.price      
        elif sold_bond < self.bonds[Bond.symbol]:
            bond_sell_price = rand_b*Bond.price          
        self.cash += bond_sell_price*sold_bond
        self.bond_shares -= sold_bond
        self.bonds[Bond.symbol] -= sold_bond
        self.transactions.append(f"{sold_bond} number of {Bond.symbol} shares withdrawn from the Bonds account.")
        return self.bonds

