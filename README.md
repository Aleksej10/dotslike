# Igra dots, ili tako nesto

Strahinja Ivanovic, Darko Veizovic


## Uvod

### Abastract
> Igre na tabli sa dva igraca, kao sto su sah, go ili sogi, su najduze izucavan
> domen u istoriji vestacke inteligencije. Najjaci programi stvoreni da igraju
> ove igre su bazirani na kombiaciji sofisticiranih tehnika pretrage, adaptacija
> na specificnost igre i rucno pravljenim funkcijama evaluacije redizajniranim
> vise od nekoliko decenija od strane ljudskih eksperata u navedenim igrama.

Igra koja ce biti opisana je slabo poznata, cak je i njeno ime nepoznanica na celom
internetu (do druge strane google pretrage), pa ce se u nastavku na nju referisati
kao na igru "Dots" (iako bi pogodnije ime verovatno bilo LINES).

Tabla ove igre podseca na znak KARO u kartama. Sastoji se od 57 kvadrata koji mogu
biti oznaceni kao 'x' ili 'o'. Svaki kvadrat formiraju 4 linije, a cela tabla ih
ima ukupno 96. U svakom potezu, igrac markira jednu liniju na tabli. Ako je pri tom
formirao kvadrat (ili dva) ostaje na potezu i markira sledecu liniju, tako rekurzivno.
Pobednik je igrac koji je zauzeo vise kvadrata. Ovako formulisana, igra Dots podseca na
igru go, ali odnosi kvadrata i linija, kao i sam oblik table, prave dovoljnu razliku.


Ne postoje poznati programi koji igraju ovu igru, pa su za potrebe uporedjivanja
koriscena tri objekta:
  -  program koji nasumicno odigrava poteze
  -  klasicni alpha/beta sa intuitivnom funkcijom evaluacije
  -  ljudski ekspert

Program opisan ovde dosegao je nadljudske performanse,
pobedivsi ubedljivo sva tri objekta. Nasuprot opisanim metodama iz uvodnog dela,
program ne zna nista od igri sem njenih pravila.



## Metode


### Anatomija klasicnog "glupog" Dots programa

Stanje pozicije odredjeno je binarnim brojem od 96 cifara, gde
'1' predstavlja upisanu liniju, brojem polja obelezenih kao
'x' ili 'o' i znakom igraca na potezu (1 ili -1).
Evaluacija pozicije dobije se pravljenjem stabla, kod koga se iz
svake pozicije potezima dolazi do nove pozicije.
Dubina pretrage, iliti visina stabla, zavisi od pozicije, jer se
na primer iz pocetne pozicije u narednih 5 poteza moze dobiti vise
od 7,000,000,000 mogucih stanja na tabli, dok se pri kraju igre
dubina pretrage moze znatno uvecati zbog suzenog izbora poteza, pa tako
u poziciji koja je 10 poteza pred kraj mozemo razviti stablo do listova
i tako napraviti nesto vise od 3,500,000 cvorova.
Dodatno, alpha/beta srezivanjem izbegavamo grananje stabla u delove
gde je jedna strana verovatno potpuno dominirana drugom.
Kao funkcija evaluacije svakog od listova (ili pseudo listova) koristi
se obicna razlika izmedju obelezenih polja. Konacna evaluacija dobija se
minmax pretragom, gde u svakom nivou uzimamo maximum ili miminum u zavisnosi od
strane na potezu u tom nivou (pozitivne vrednosti su dobre za 'x', negativne za 'o').

Ni jedna od opisanih tehnika se ne koristi u ovom programu, iako bi neko od njih
verovatno dovele do boljih performansi. Bez obzira na to, program je fokusiran
na cistom ucenju iz igranja protiv samog sebe.

### Dots

Umesto alpha/beta pretrage, u ovom programu koristi se Monte Carlo Tree Search.
Svaki od cvorova predstavljen je svojom pozijom na tabli, broju obilaska i sumom
estimacija, kao i inicijalnom estimacijom neuronske mreze koja se pamti zbog
kasnijeg podesavanja same mreze. MTCS implementiran u programu je veoma nalik
klasicnom, s tim da se umesto rollout-a, gde se kao evaluacije pozicija uzimaju
srednje vrednosti dobijene kao rezultat simulacija iz date pozicije, evaluacija
dobija pomocu neuronske mreze koja vraca sanse za pobedu iz te pozicije, od -1 do 1,
gde 1 znaci da 'x' sigurno dobija, dok -1 donisi sigurnu pobedu za 'o' stranu.
Prilikom formiranja stabla, USB1 algoritmom pravi se balans izmedju istrazivanja novih
grana i dubljeg ptretrazivanja obecavajucih. Na kraju pretrage program pohlepno
bira najjaci potez u odnosu na broj obilaska potencijalnih poteza i njihovih sansi za
donosenje pobede.

Po zavrsetku igre, unutrasnja stanja neuronske mreze se azuriraju u skladu sa
ishodom igre, kao i odnosa inicijalne estimacije pozicije izmedju bliskih cvorova.
Mreza se modifikuje kako bi smanjila razliku izmedju estimacije cvora i srednjih vredosti
estimacija dva cvora posle njega, kao i razliku inicijalne estimacije verovatnoce i
verovatnoce dobijene MTCS pretragom. Ako estimacije pri svakom odigranom potezu nasumicno
inicijalizovane mreze predstavimo u grafu, mozemo ocekivati da dobijemo isprekidanu
cik-cak liniju koja ukazuje na to da program nema pojma sta se desava, u jednom potezu
je siguran da pobedjuje, vec u sledecem gubi. Azuriranje neuronske mreze na ovaj nacin
izgladjuje taj grafik nakon svake iteracije, sto reflektuje "shvatanje" pozicije.
Za resavanje ove igre deterministickim metodama bio bi potreban obilazak 96! pozicije,
sto je, 991677934870949689209571401541893801158183648651267795444376054838492222809091499987689476037000748982075094738965754305639874560000000000000000000000.
Ovakvom metodom za svaku poziciju imali bi smo konacnu evaluaciju. MTCS pristupom,
uz mnogo manje obilazaka, verovatnoca ishoda konvergira toj evaluaciji.

## Deo sa grafovima

Za sad prazan.
Specifikacija hardvera na kojoj je mreza trenirana:
  -  GPU: 1xTesla K80, 2496 CUDA cores, 12GB GDDR5 VRAM
  -  CPU: 1xsingle core hyper threaded Xeon Processors @2.3Ghz, 45MB Cache



## Osvrt i predlozi poboljsanja

Kao prostor za poboljsanje efikasnosti programa, kao i brzina njegovog
izvrsavanja i treniranja neuronske mreze, mozemo kao ocigledan pocetak uzeti
izbor jezika u kome je program pisan. Bez iole kontroverzije mozemo reci
da je python spor, pa bi se implementacijom istog programa u npr. C-u postiglo
ogromno poboljsanje performansi. Medjutim, zbog skromnih karakteristika nasih
racunara, neuronska mreza je trenirana na [google colab-u](https://colab.research.google.com),
koji za sada podrzava samo python. Sa druge strane, sa bibliotekama za python
za rad sa vestackom inteligencijom kao sto su [tensorflow](https://tensorflow.org) i [keras](keras.io)
u zamenu za perormanse dobijate udobnost pri implementaciji i lakocu prenosenja
ideja u kod.
Kao konkretna oboljsanja koda mogu se dodati pomenute heuristike (alpha/beta i ostale).
Jedna od stvari koja je mogla biti iskoriscena je binarna priroda ishoda igre.
Sa 57 mogucih polja, jasno je da se igra moze zavrsiti samo porazom ili pobedom, pa
se predstavljanjem verovatnoca ishoda moglo bolje predstaviti u odnosu na pobedu jedne strane.
Samim tim verovatnoce bi se kretale od 0 do 1, sto bi omogucilo primenu drugacije
funkcije aktivacije (sigmoid umesto tanh) koja bi verovatno doprinela brzem konvergiranju
optimumu.
Takodje, jos jedna stvar specificna za igru koja se mogla iskoristiti je simetricnost table.
Znajuci to, pre vrsenja estimacije tabla kao ulazni parametar u neuronsku mrezu bi
bila proizvoljno rotirana, cime bi se smanjila mogucnost overfitovanja i generalno
ubrzao postupak treniranja.


## Literatura

Svaki deo koda bio je rucno dizajniran, bez kopiranja sa interneta.
Ocekivano je naci slicnosti u ociglednim delovima, kao sto je siroko poznati MCTS
ili kompajliranje Keras modela.
Veliki deo ideja uzet je iz [AlphaZero](http://deepmind.com/blog/alphazero-shedding-new-light-grand-games-chess-shogi-and-go/) papira koji je objavila
kompanija [DeepMind](http://deepmind.com).
Takodje, kao smernica je posluzio [Lc0](https://lczero.org) a pomoc u
shvatanju pojedinjenih koncepata je dobijena od strane dobrih ljudi sa
[leela chess zero discord ceta](https://discord.gg/pKujYxD).
