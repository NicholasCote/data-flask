import ldap
import getpass
con = ldap.initialize('ldaps://cit.ucar.edu')
user_dn = r"ncote@ucar.edu"
password = getpass.getpass()
  
try:
    con.simple_bind_s(user_dn, password)
    res = con.search_s("OU=CISL,OU=Divisions,DC=cit,DC=ucar,DC=edu", ldap.SCOPE_SUBTREE, '(objectClass=*)')
    for i in res:
        #if i[1].get('memberOf') is not None:
            #print(str(i[1].get('sAMAccountName')) + ' - ' + str(i[1].get('memberOf')))
        if 'katelynw' in str(i[1].get('sAMAccountName')):
            print(str(i[1].get('memberOf')))

except Exception as e:
    print(e)