/* 1.	Visualizzare i clienti (customers) in ordine alfabetico*/
select *
from Customers
order by ContactName;

/* 2.	Visualizzare i clienti che non hanno il fax*/
select *
from Customers
where Fax is null;

/* 3.	Selezionare i nomi dei clienti (CompanyName) che iniziano con le lettere P, Q, R, S*/
select CompanyName
from Customers
where CompanyName like 'P%' or CompanyName like 'Q%' or CompanyName like 'R%' or CompanyName like 'S%';

select CompanyName
from Customers
where CompanyName regexp '^([PQRS].*)$';

/* 4.	Visualizzare Nome e Cognome degli impiegati assunti (HireDate) dopo il '1993-05-03' e aventi titolo di “Sales Representative”*/
select *
from Employees
where HireDate > '1993-05-03' and Title='Sales Representative';

/* 5.	Selezionare la lista dei prodotti non sospesi (attributo discontinued), visualizzando 
anche il nome della categoria di appartenenza*/
select Products.*, Categories.CategoryName
from Products, Categories
where not Discontinued and Products.CategoryID = Categories.CategoryID
order by ProductID;

select Products.*, Categories.CategoryName
from Products join Categories on (Products.CategoryID = Categories.CategoryID)
where Discontinued = 0 and Products.CategoryID = Categories.CategoryID
order by ProductID;
/* 6. Selezionare gli ordini relativi al cliente ‘Ernst Handel’ (CompanyName)*/
select Customers.CompanyName, Orders.*
from Orders join Customers on (Orders.CustomerID = Customers.CustomerID)
where Customers.CompanyName = 'Ernst Handel';

/* 7.	Selezionare il nome della società e il telefono dei corrieri (shippers) che hanno consegnato 
ordini nella città di ‘Rio de Janeiro’ (ShipCity in orders)
N.B. nella tabella orders l'id corriere è l'attributo ShipVia*/
select distinct ShipperID, CompanyName, Phone
from Shippers join Orders on (Orders.ShipVia = Shippers.ShipperID)
where ShipCity = 'Rio de Janeiro';

/* 8.	Selezionare gli ordini (OrderID, OrderDate, ShippedDate) per cui la spedizione (ShippedDate)
è avvenuta entro 30 giorni dalla data dell’ordine (OrderDate)*/

/* 9.	Selezionare l’elenco dei prodotti che hanno un costo compreso tra 18 e 50, ordinando il risultato
in ordine di prezzo crescente */
select *
from Products
where UnitPrice between 18 and 50
order by UnitPrice;

/* 10.	Selezionare tutti i clienti (CustomerID, CompanyName) che hanno ordinato il prodotto 'Chang'*/
select distinct C.CustomerID, CompanyName
from Customers C,Orders O, `Order Details` OD,Products P 
where C.CustomerID = O.CustomerID and OD.OrderID = O.OrderID and OD.ProductID = P.ProductID
and ProductName = 'Chang';

/* 11.	Selezionare i clienti che non hanno mai ordinato prodotti di categoria ‘Beverages’*/

/* 12.	Selezionare il prodotto più costoso*/

/* 13.	Visualizzare l’importo totale di ciascun ordine fatto dal cliente 'Ernst Handel' (CompanyName)*/

/* 14.	Selezionare il numero di ordini ricevuti in ciascun anno */

/* 15.	Visualizzare per ogni impiegato (EmployeeID, LastName, FirstName) il numero di clienti distinti serviti per ciascun paese (Country),
visualizzando il risultato in ordine di impiegato e di paese*/

/* 16.	Visualizzare per ogni corriere il numero di consegne effettuate, compresi i dati dei 
corrieri che non hanno effettuato nessuna consegna */

/* 17.	Visualizzare i fornitori (SupplierID, CompanyName) che forniscono un solo prodotto */

/* 18.	Visualizzare tutti gli impiegati che sono stati assunti dopo Margaret Peacock */

/* 19.	Visualizzare gli ordini relativi al prodotto più costoso */
                                       
/* 20.	Visualizzare i nomi dei clienti per i quali l’ultimo ordine è relativo al 1998  */

/* 21.	Contare il numero di clienti che non hanno effettuato ordini */
                         
/* 22.	Visualizzare il prezzo minimo, massimo e medio dei prodotti per ciascuna categoria */

/* 23.	Selezionare i prodotti che hanno un prezzo superiore al prezzo medio dei prodotti forniti dallo stesso fornitore */
                    
/* 24.	Visualizzare, in ordine decrescente rispetto alla quantità totale venduta, i prodotti che hanno venduto una quantità 
totale superiore al prodotto ‘Chai’ */

/* 25.	Visualizzare il nome dei clienti che hanno fatto almeno due ordini di importo superiore a 15000 */

/* 26.	Individuare i codici dei clienti che hanno fatto un numero di ordini pari a quello del cliente 'Eastern Connection' */

/* 27. Visualizzare il numero di ordini ricevuti nel 1997 e di importo superiore a 10000*/

/* 28. Visualizzare i corrieri che hanno consegnato un numero di ordini superiore al numero di ordini consegnati da Speedy Express. */

/* 29. Visualizzare i clienti che hanno ordinato prodotti di tutte le categorie */

                     

